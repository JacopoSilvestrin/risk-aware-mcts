import numpy as np
import random
from envs.envs import Env

CATEGORY_MAP = {
    'Dairy': 0, 'Bakery': 1, 'Produce': 2, 'Snacks': 3,
    'Pantry': 4, 'Frozen': 5, 'Raw Meat': 6, 'Cleaning': 7, 'Household': 8
}
TEMPERATURE_MAP = {
    'Ambient': 0, 'Refrigerated': 1, 'Frozen': 2
}

# Cost constants — tune these to calibrate accident signal vs. packing noise.
# Illegal move reduced so it doesn't completely swamp accident variance during rollouts.
# Crush/spill raised so a single accident is visible on the scale of packing-reward differences.
#
# Two normalization stages, both derived from these raw numbers:
#   1. Per-step (below): a single step's cost is at most 1. The worst case for
#      num_bags=5 is the last item landing on top of 4 already-packed fragile items in the
#      same bag and crushing all of them (4 * 50.0 = 200, the raw CRUSH_COST above), so every
#      raw cost below is divided by that bound.
#   2. Per-episode (BinPackingEnv.__init__, which divides these by self.H): assuming every
#      step hits that same per-step worst case (it can't, in practice, since a bag can only be
#      "broken into" with 4 pre-existing items once), a full H-step episode's cumulative cost
#      is then at most ~1 too. If num_bags or the worst-case scenario above ever changes,
#      recompute this bound.
_STEP_COST_NORMALIZATION = 200.0

ILLEGAL_MOVE_COST  = 100.0 / _STEP_COST_NORMALIZATION  # volume/weight constraint violation
CRUSH_COST         = 50.0 / _STEP_COST_NORMALIZATION    # per fragile item crushed
SPILL_COST         = 40.0 / _STEP_COST_NORMALIZATION    # base cost per spill event
CONTAMINATION_COST = 10.0 / _STEP_COST_NORMALIZATION    # added per spill-vulnerable item already in bag when spill occurs
BAG_OPEN_COST      = 10.0 / _STEP_COST_NORMALIZATION    # charged once when the first item is placed in a previously empty bag

class Item:
    def __init__(self, id, name, est_weight_g, est_volume_cc, crush_score, category, temperature, spill_risk, spill_vulnerable, orientation_sensitive, is_crushed=False, risk_level=0.0, is_spilled=False, spill_risk_level=0.0):
        self.id = id
        self.name = name
        self.weight = est_weight_g
        self.volume = est_volume_cc
        self.crush_score = crush_score
        self.category = category
        self.temperature = temperature
        self.spill_risk = spill_risk
        self.spill_vulnerable = spill_vulnerable
        self.orientation_sensitive = orientation_sensitive
        self.is_crushed = is_crushed
        self.risk_level = risk_level
        self.is_spilled = is_spilled
        self.spill_risk_level = spill_risk_level

    def copy(self):
        return Item(
            self.id, self.name, self.weight, self.volume, self.crush_score,
            self.category, self.temperature, self.spill_risk, self.spill_vulnerable,
            self.orientation_sensitive, self.is_crushed, self.risk_level,
            self.is_spilled, self.spill_risk_level
        )

    def get_features(self):
        category_id = CATEGORY_MAP.get(self.category, -1)
        temp_id = TEMPERATURE_MAP.get(self.temperature, -1)
        return (
            self.weight,
            self.volume,
            self.crush_score,
            category_id,
            temp_id,
            int(self.spill_risk),
            int(self.spill_vulnerable),
            int(self.orientation_sensitive),
            int(self.is_crushed),
            self.risk_level,
            int(self.is_spilled),
            self.spill_risk_level,
        )

    def __repr__(self):
        return f"Item({self.name}_{self.id})"

class Bag:
    MAX_VOLUME = 5000
    MAX_WEIGHT = 5000

    def __init__(self):
        self.volume_used = 0
        self.weight_used = 0
        self.items = []
        self.fragile_count = 0
        self.spill_risk_count = 0
        self.spill_vulnerable_count = 0
        self.orientation_sensitive_count = 0
        self.top_level_risk_score = 0
        self.total_items = 0
        self.spilled_count = 0
        
    def copy(self):
        new_bag = Bag()
        new_bag.volume_used = self.volume_used
        new_bag.weight_used = self.weight_used
        new_bag.items = [it.copy() for it in self.items]
        new_bag.fragile_count = self.fragile_count
        new_bag.spill_risk_count = self.spill_risk_count
        new_bag.spill_vulnerable_count = self.spill_vulnerable_count
        new_bag.orientation_sensitive_count = self.orientation_sensitive_count
        new_bag.top_level_risk_score = self.top_level_risk_score
        new_bag.total_items = self.total_items
        new_bag.spilled_count = self.spilled_count
        return new_bag

    def get_features(self):
        crushed_count = sum(1 for it in self.items if it.is_crushed)
        total_risk = sum(it.risk_level for it in self.items)
        total_spill_risk = sum(it.spill_risk_level for it in self.items)
        return (
            self.volume_used,
            self.weight_used,
            crushed_count,
            total_risk,
            self.fragile_count,
            self.spill_risk_count,
            self.spill_vulnerable_count,
            self.orientation_sensitive_count,
            self.spilled_count,
            total_spill_risk,
        )

    def __repr__(self):
        return f"Bag(features={self.get_features()})"


# List of items from grocery_list.json.
# Paper Towels (vol=8000 cc) and Toilet Paper (vol=6000 cc) removed — both exceed
# Bag.MAX_VOLUME=5000 and could never be legally placed, forcing a 100% illegal-move rate.
ITEMS_DATA = [
    # Item(id, name, est_weight_g, est_volume_cc, crush_score, category, temperature, spill_risk, spill_vulnerable, orientation_sensitive)
    Item(1,  "Whole Milk",           2500, 2000, 5,  "Dairy",     "Refrigerated", True,  False, True),
    Item(2,  "Bleach",               2500, 2000, 1,  "Cleaning",  "Ambient",      True,  False, True),
    Item(3,  "Loaf of Bread",         200, 3000, 9,  "Bakery",    "Ambient",      False, True,  False),
    Item(4,  "Cheddar Cheese",        200,  600, 5,  "Dairy",     "Refrigerated", False, False, False),
    Item(5,  "Ground Beef",           800, 1200, 4,  "Raw Meat",  "Refrigerated", True,  False, True),
    Item(6,  "Bag of Potato Chips",   200, 2000, 9,  "Snacks",    "Ambient",      False, True,  False),
    Item(7,  "Carton of Eggs",        800, 1200, 10, "Dairy",     "Refrigerated", True,  True,  True),
    Item(8,  "Apples",                800, 1200, 9,  "Produce",   "Ambient",      False, True,  False),
    Item(9,  "Frozen Pizza",          800, 1200, 3,  "Frozen",    "Frozen",       False, False, False),
    Item(11, "Chicken Breast",        800, 1200, 4,  "Raw Meat",  "Refrigerated", True,  False, True),
    Item(12, "Orange Juice",         1750, 1800, 6,  "Dairy",     "Refrigerated", True,  False, True),
    Item(13, "Yogurt",                500,  500, 5,  "Dairy",     "Refrigerated", False, False, False),
    Item(14, "Pasta",                 500, 1000, 3,  "Pantry",    "Ambient",      False, False, False),
    Item(15, "Canned Tomatoes",       400,  400, 1,  "Pantry",    "Ambient",      False, False, False),
    Item(16, "Bananas",               500, 1500, 8,  "Produce",   "Ambient",      False, True,  False),
    Item(17, "Strawberries",          300,  700, 9,  "Produce",   "Refrigerated", False, True,  False),
    Item(18, "Ice Cream",             600, 1500, 4,  "Frozen",    "Frozen",       False, False, True),
    Item(19, "Frozen Vegetables",     400,  800, 2,  "Frozen",    "Frozen",       False, False, False),
    Item(20, "Pork Chops",            700,  900, 3,  "Raw Meat",  "Refrigerated", True,  False, False),
    Item(21, "Dish Soap",             500,  800, 2,  "Cleaning",  "Ambient",      True,  False, True),
    Item(22, "Laundry Detergent",    2000, 2200, 1,  "Cleaning",  "Ambient",      True,  False, True),
    Item(24, "Crackers",              250, 2000, 9,  "Snacks",    "Ambient",      False, True,  False),
    Item(25, "Granola Bars",          300, 1000, 6,  "Snacks",    "Ambient",      False, False, False),
]

DEBUG_ITEMS_DATA = [
    ITEMS_DATA[6],  # Carton of Eggs (id=7, Fragile)
    ITEMS_DATA[0],  # Whole Milk (id=1, Heavy)
    ITEMS_DATA[9],  # Chicken Breast (id=11)
    ITEMS_DATA[3],  # Cheddar Cheese (id=4)
]

# Small item pool for reduced-stochasticity experiments: item arrival is still
# randomly sampled (unlike DEBUG_ITEMS_DATA's fixed cyclic sequence), just from
# a smaller catalog, to cut outcome variance while keeping randomness.
REDUCED_ITEMS_DATA = [
    ITEMS_DATA[6],   # Carton of Eggs (id=7) — fragile (crush_score=10), spill risk + spill vulnerable
    ITEMS_DATA[0],   # Whole Milk (id=1) — heavy, spill risk
    ITEMS_DATA[9],   # Chicken Breast (id=11) — raw meat, spill risk
    ITEMS_DATA[3],   # Cheddar Cheese (id=4) — small, no spill risk
    ITEMS_DATA[14],  # Bananas (id=16) — fragile (crush_score=8), spill vulnerable, no spill risk of its own
]

class BinPackingEnv(Env):
    def __init__(self, num_bags=5, num_items_to_pack=5, gamma=1, debug=False, reduced=False):
        # We don't call super().__init__(mdp, H) because we don't use an explicit MDP dict.
        self.num_bags = num_bags
        self.num_items_to_pack = num_items_to_pack
        self.gamma = gamma
        self.H = num_items_to_pack # Horizon is exactly the number of items to pack

        # Per-episode cost scale: divide the (already per-step-normalized) module-level
        # costs by H, so a full episode's cumulative cost is at most ~1 (see the comment
        # above the module-level constants). Instance attributes, not module constants,
        # because H is configurable per BinPackingEnv instance.
        self.illegal_move_cost  = ILLEGAL_MOVE_COST / self.H
        self.crush_cost         = CRUSH_COST / self.H
        self.spill_cost         = SPILL_COST / self.H
        self.contamination_cost = CONTAMINATION_COST / self.H
        self.bag_open_cost      = BAG_OPEN_COST / self.H

        self.items_data = ITEMS_DATA
        self.debug = debug
        self.debug_items = DEBUG_ITEMS_DATA
        self.reduced = reduced
        self.reduced_items = REDUCED_ITEMS_DATA

    def available_actions(self, state):
        # Actions are bag indices 0 to num_bags-1
        return list(range(self.num_bags))

    def _get_rl_vector(self, bags, current_item):
        vec = [current_item.id]
        for bag in bags:
            vec.extend(bag.get_features())
        return tuple(vec)

    def sample_initial_state(self):
        # In a generic MDP, we sample the initial state.
        # Here we randomly select the first item for this episode.
        if self.debug:
            current_item = self.debug_items[0]
        elif self.reduced:
            current_item = random.choice(self.reduced_items)
        else:
            current_item = random.choice(self.items_data)
        
        # State contains lists/tuples so it can be hashed (ERM MCTS uses str representation to hash)
        bags = tuple(Bag() for _ in range(self.num_bags))
        
        extended_state = {
            "state": self._get_rl_vector(bags, current_item),
            "bags": bags,
            "current_item": current_item,
            "t": 0
        }
        return extended_state

    def step(self, extended_state, a):

        # Read properties from current state
        state_t, t = extended_state["state"], extended_state["t"]
        bags = list(bag.copy() for bag in extended_state["bags"])
        item = extended_state["current_item"]
        
        cost_t = 0
        terminated = False

        bag = bags[a]
        
        illegal_move = False
        
        # Check volume/weight constraints and if bag is already broken
        if bag.volume_used + item.volume > bag.MAX_VOLUME or \
             bag.weight_used + item.weight > bag.MAX_WEIGHT:
             illegal_move = True
             cost_t += self.illegal_move_cost

        if not illegal_move:
             # Make state transition
             bag.volume_used += item.volume
             bag.weight_used += item.weight
             bag.total_items += 1
             if bag.total_items == 1:
                 cost_t += self.bag_open_cost

             # Risk contribution from putting items on top of fragile/sensitive items
             # 1. Update risk for each fragile item ALREADY in the bag
             for packed_item in bag.items:
                 if packed_item.crush_score >= 8 and not packed_item.is_crushed:
                     # Increase risk based on weight of the new item being placed on top
                     if item.crush_score <= 2:
                         packed_item.risk_level += item.weight / 500.0
                     else:
                         packed_item.risk_level += item.weight / 1000.0

                     # Check for crushing
                     fail_prob = min(packed_item.risk_level * 0.05, 1.0) #0.003
                     if np.random.rand() < fail_prob:
                         packed_item.is_crushed = True
                         cost_t += self.crush_cost

             # 2. Update spill risk for each spill-risk item ALREADY in the bag.
             # Adding a new item josttles the bag; the tighter the bag, the higher the chance of
             # tipping an orientation-sensitive container.
             squeeze_factor = bag.volume_used / Bag.MAX_VOLUME
             for packed_item in bag.items:
                 if packed_item.spill_risk and not packed_item.is_spilled:
                     # Orientation-sensitive items (e.g. open bottles) accumulate risk faster
                     if packed_item.orientation_sensitive:
                         packed_item.spill_risk_level += (item.weight / 500.0) * squeeze_factor
                     else:
                         packed_item.spill_risk_level += (item.weight / 1000.0) * squeeze_factor

                     fail_prob = min(packed_item.spill_risk_level * 0.05, 1.0) #0.003
                     if np.random.rand() < fail_prob:
                         packed_item.is_spilled = True
                         bag.spilled_count += 1
                         cost_t += self.spill_cost
                         # Extra contamination cost for each spill-vulnerable item already in the bag
                         contaminated = sum(
                             1 for it in bag.items
                             if it.spill_vulnerable and not it.is_crushed and it is not packed_item
                         )
                         cost_t += self.contamination_cost * contaminated

             # 3. Add new item to bag
             bag.items.append(item.copy())

             # Update counts
             if item.crush_score >= 8:
                 bag.fragile_count += 1
             if item.spill_risk:
                 bag.spill_risk_count += 1
             if item.spill_vulnerable:
                 bag.spill_vulnerable_count += 1
             if item.orientation_sensitive:
                 bag.orientation_sensitive_count += 1
                 
        # Build next extended state
        next_t = t + 1
        if self.debug:
            if next_t < len(self.debug_items):
                next_item = self.debug_items[next_t]
            else:
                # Cycle if we exceed debug items
                next_item = self.debug_items[next_t % len(self.debug_items)]
        elif self.reduced:
            next_item = random.choice(self.reduced_items)
        else:
            next_item = random.choice(self.items_data)
        
        next_state = {
                "state": self._get_rl_vector(bags, next_item), 
                "bags": tuple(bags),
                "current_item": next_item,
                "t": next_t
            }
        
        # Horizon H reached
        if next_t >= self.H:
            terminated = True
            
        return next_state, cost_t, terminated

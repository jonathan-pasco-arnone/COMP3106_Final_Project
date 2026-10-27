def single_fuzzification(x,a,b,c,d):
    # Check validity of values
    # If a is None, then b has to be as well (same with c and d)
    if (a is None) != (b is None) or (c is None) != (d is None):
        return None

    # Full infinity on both sides
    if a is None and c is None:
        return 1

    # Left shoulder: [None, None, c, d]
    if a is None:  
        if x <= c:
            return 1
        # Already know x > c from previous if
        if x <= d:
            return (d-x)/(d-c)
        return 0

    # Right shoulder: [a, b, None, None]
    if c is None:
        if x <= a:
            return 0
        if x <= b:
            return (x-a)/(b-a)
        return 1

    # Core fuzzy membership function definition
    if x <= a:
        return 0
    if x <= b:
        return (x-a)/(b-a)
    if x <= c:
        return 1
    if x <= d:
        return (d-x)/(d-c)
    return 0
    

def full_fuzzification(fuzzy_set, x):
    scores = []
    for name, (a,b,c,d) in fuzzy_set.items():
        scores.append({name: single_fuzzification(x,a,b,c,d)})
        
    return scores

class fuzzySet:
    # Isn't actually used in rules.json as it is not needed for this dataset, but could use NOT if needed.
    @staticmethod
    def NOT(x):
        return 1-x

    @staticmethod
    def AND(x,y):
        return min(x,y)
    
    @staticmethod
    def OR(x,y):
        return max(x,y)
        
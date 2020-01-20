'''

### Layer2:

    Help + weak > Help + strong > Not Help + weak > Not Help + strong

- Help: reputation increase is proportional to cost Index (cost +, reputationIncrease +)
- Not Help: amt. of reputation decrease is inversely Proportional to cost Index (cost +, reputationDecrease - )

'''

class GetReputationChange:
    def __init__(self, reputBoostParam, reputHurtParam):
        self.reputBoostParam = reputBoostParam  # positive
        self.reputHurtParam = reputHurtParam  # negative

    def __call__(self, agentAbilityScore, helpingCost, helped):
        costAbilityRatio = helpingCost / agentAbilityScore
        if helped:
            reputationChange = self.reputBoostParam * costAbilityRatio
        else:
            reputationChange = self.reputHurtParam * max(0, (1 - costAbilityRatio))

        print("helped", helped, "reputationChange", reputationChange)
        return reputationChange



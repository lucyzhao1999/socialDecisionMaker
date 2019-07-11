'''

### Layer2:

    Help + weak > Help + strong > Not Help + weak > Not Help + strong

- Help: reputation increase is proportional to cost Index (cost +, reputationIncrease +)
- Not Help: amt. of reputation decrease is inversely Proportional to cost Index (cost +, reputationDecrease - )

'''

class GetReputationChange:
    def __init__(self, reputBoostIndex, reputHurtIndex):
        self.reputBoostIndex = reputBoostIndex  # positive
        self.reputHurtIndex = reputHurtIndex  # negative

    def __call__(self, agentAbilityScore, helpingCost, helped):
        costIndex = helpingCost / agentAbilityScore
        if helped:
            reputationChange = self.reputBoostIndex * costIndex
        else:
            reputationChange = self.reputHurtIndex * (1 - costIndex)

        print("helped", helped, "reputationChange", reputationChange)
        return reputationChange



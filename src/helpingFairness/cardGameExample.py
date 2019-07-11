'''

### AllocationGame:

- Rules:
    * 0 -9 cards: #points = #cards
    * 10-19 cards: #points = #cards * 2
    * ...

- one player request a certain amount of cards, if help, helper #cards(new) = #cards(old) - #cardsRequested

'''

def getAgentScore(agentCardCount):
    groupIndex = agentCardCount // 10
    score = (groupIndex + 1) * agentCardCount
    return score


class GetBaseUtility:
    def __init__(self, costAbilityRatio, getAgentScore):
        self.costAbilityRatio = costAbilityRatio
        self.getAgentScore = getAgentScore

    def __call__(self, helperCardCount, receiverCardCount, cardRequested, helped):
        receiverNewCount = receiverCardCount + cardRequested
        helpingReward = self.getAgentScore(receiverNewCount) - self.getAgentScore(receiverCardCount)
        print("helpingReward", helpingReward)

        helperNewCount = helperCardCount - cardRequested
        helpingCost = abs(self.getAgentScore(helperNewCount) - self.getAgentScore(helperCardCount))
        print("helpingCost", helpingCost)

        agentAbilityScore = self.getAgentScore(helperCardCount)
        print("agentAbilityScore", agentAbilityScore)

        cost = helpingCost / agentAbilityScore * self.costAbilityRatio

        helpingIndex = int(helped)
        utility = (helpingReward - cost) * helpingIndex
        return utility
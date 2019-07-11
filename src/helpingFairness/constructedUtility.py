'''

### Layer3:

constructedUtility = baseUtility + reputationChange

'''
import math

class GetConstructedUtility:
    def __init__(self, getBaseUtility, getReputationChange):
        self.getBaseUtility = getBaseUtility
        self.getReputationChange = getReputationChange

    def __call__(self, agentAbilityScore, helpingCost, helped):
        baseUtility = self.getBaseUtility(helpingCost, helped)
        reputationChange = self.getReputationChange(agentAbilityScore, helpingCost, helped)
        constructedUtility = baseUtility + reputationChange
        return constructedUtility


class GetConstructedHelpingProb:
    def __init__(self, beta, getConstructedUtility):
        self.beta = beta
        self.getConstructedUtility = getConstructedUtility

    def __call__(self, agentAbilityScore, helpingCost):
        helpedUtility = self.getConstructedUtility(agentAbilityScore, helpingCost, helped=True)
        helpProb = math.exp(helpedUtility * self.beta)

        noHelpUtility = self.getConstructedUtility(agentAbilityScore, helpingCost, helped=False)
        noHelpProb = math.exp(noHelpUtility * self.beta)

        helpProbNormalized = helpProb / (helpProb + noHelpProb)

        return helpProbNormalized
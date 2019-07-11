'''

### Layer1:
Base Utility of helping:

    Helped -> utility= Reward - costOfHelping
    Didn't Help -> utility = 0

'''

import math


class GetBaseUtility:
    def __init__(self, helpingReward):
        self.helpingReward = helpingReward

    def __call__(self, helpingCost, helped):
        helpingIndex = int(helped)
        utility = (self.helpingReward - helpingCost) * helpingIndex
        return utility


class GetBaseHelpingProb:
    def __init__(self, beta, getBaseUtility):
        self.beta = beta
        self.getBaseUtility = getBaseUtility

    def __call__(self, helpingCost):
        helpedUtility = self.getBaseUtility(helpingCost, helped=True)
        helpProb = math.exp(helpedUtility * self.beta)

        noHelpUtility = self.getBaseUtility(helpingCost, helped=False)
        noHelpProb = math.exp(noHelpUtility * self.beta)

        helpProbNormalized = helpProb / (helpProb + noHelpProb)
        return helpProbNormalized


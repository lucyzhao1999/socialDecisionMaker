import sys
import os
sys.path.append(os.path.join('..', 'src', 'helpingFairness'))

from baseUtility import GetBaseUtility, GetBaseHelpingProb
from reputationChange import GetReputationChange
from constructedUtility import GetConstructedUtility, GetConstructedHelpingProb


def main():
    helpingReward = 0
    getBaseUtility = GetBaseUtility(helpingReward)

    beta = 0.008
    getBaseHelpingProb = GetBaseHelpingProb(beta, getBaseUtility)
    helpingCost = 100
    baseActionProb = getBaseHelpingProb(helpingCost)

    reputBoostIndex = 100
    reputHurtIndex = -150
    getReputationChange = GetReputationChange(reputBoostIndex, reputHurtIndex)

    getConstructedUtility = GetConstructedUtility(getBaseUtility, getReputationChange)
    getConstructedHelpingProb = GetConstructedHelpingProb(beta, getConstructedUtility)

    agentAbilityScore = 300
    constructedActionProb = getConstructedHelpingProb(agentAbilityScore, helpingCost)

    print("baseActionProb", baseActionProb)
    print("constructedActionProb", constructedActionProb)

if __name__ == '__main__':
    main()


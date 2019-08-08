import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import matplotlib.pyplot as plt
import numpy as np

from src.helpingFairness.baseUtility import GetBaseUtility, GetBaseHelpingProb
from src.helpingFairness.reputationChange import GetReputationChange
from src.helpingFairness.constructedUtility import GetConstructedUtility, GetConstructedHelpingProb
from src.helpingFairness.wrappers import GetReputationBoostParam, GetReputationHurtParam

# def main():
#
#     helpingReward = 0
#     getBaseUtility = GetBaseUtility(helpingReward)
#
#     beta = 0.01
#     getBaseHelpingProb = GetBaseHelpingProb(beta, getBaseUtility)
#     helpingCost = 100
#     agentAbilityScore = np.linspace(20, 1000, 50)
#     baseActionProbList = [getBaseHelpingProb(score, helpingCost) for score in agentAbilityScore]
#
#     reputBoostIndex = 100
#     reputHurtIndex = -500
#     getReputationChange = GetReputationChange(reputBoostIndex, reputHurtIndex)
#
#     getConstructedUtility = GetConstructedUtility(getBaseUtility, getReputationChange)
#     getConstructedHelpingProb = GetConstructedHelpingProb(beta, getConstructedUtility)
#
#     constructedActionProb = [getConstructedHelpingProb(score, helpingCost) for score in agentAbilityScore]
#
#     plt.plot(agentAbilityScore, baseActionProbList, color='blue', label='baseHelpProb')
#     plt.plot(agentAbilityScore, constructedActionProb, color='grey', label='constructedHelpProb')
#     plt.legend(loc='upper left')
#     plt.ylabel("help prob")
#     plt.xlabel("ability score")
#     plt.ylim(-0.1, 1.1)
#     plt.show()

def main():
    helpingReward = 0
    getBaseUtility = GetBaseUtility(helpingReward)

    beta = 0.01
    getBaseHelpingProb = GetBaseHelpingProb(beta, getBaseUtility)
    helpingCost = 100
    agentAbilityScore = np.linspace(20, 1000, 50)
    baseActionProbList = [getBaseHelpingProb(score, helpingCost) for score in agentAbilityScore]

    reputBoostIndex = 100
    reputHurtIndex = -500
    getReputationChange = GetReputationChange(reputBoostIndex, reputHurtIndex)

    getConstructedUtility = GetConstructedUtility(getBaseUtility, getReputationChange)
    getConstructedHelpingProb = GetConstructedHelpingProb(beta, getConstructedUtility)

    constructedActionProb = [getConstructedHelpingProb(score, helpingCost) for score in agentAbilityScore]

    plt.plot(agentAbilityScore, baseActionProbList, color='blue', label='baseHelpProb')
    plt.plot(agentAbilityScore, constructedActionProb, color='grey', label='constructedHelpProb')
    plt.legend(loc='upper left')
    plt.ylabel("help prob")
    plt.xlabel("ability score")
    plt.ylim(-0.1, 1.1)
    plt.show()
    
##############################################################
    helpingReward = 0
    getBaseUtility = GetBaseUtility(helpingReward)

    beta = 0.01
    getBaseHelpingProb = GetBaseHelpingProb(beta, getBaseUtility)
    helpingCost = 100
    agentAbilityScore = 300
    baseActionProb = getBaseHelpingProb(agentAbilityScore, helpingCost)

    probOfPublicList = np.linspace(0, 1, 50)
    distributionParam = 2
    boostMultiplier = 100
    hurtMultiplier = -500
    getReputationBoostParam = GetReputationBoostParam(distributionParam, boostMultiplier)
    boostParamList = getReputationBoostParam(probOfPublicList)

    getReputationHurtParam = GetReputationHurtParam(distributionParam, hurtMultiplier)
    hurtParamList = getReputationHurtParam(probOfPublicList)

    getReputationChangeList = [GetReputationChange(reputBoostIndex, reputHurtIndex)
                           for reputBoostIndex, reputHurtIndex in zip(boostParamList, hurtParamList)]

    getConstructedUtilityList = [GetConstructedUtility(getBaseUtility, getReputationChange)
                             for getReputationChange in getReputationChangeList]

    getConstructedHelpingProbList = [GetConstructedHelpingProb(beta, getConstructedUtility)
                                 for getConstructedUtility in getConstructedUtilityList]

    constructedActionProbList = [getConstructedHelpingProb(agentAbilityScore, helpingCost)
                                 for getConstructedHelpingProb in getConstructedHelpingProbList]

    baseActionProbList = [baseActionProb] * len(probOfPublicList)

    plt.plot(probOfPublicList, baseActionProbList, color='blue', label='baseHelpProb')
    plt.plot(probOfPublicList, constructedActionProbList, color='grey', label='constructedHelpProb')
    plt.legend(loc='upper left')
    plt.ylabel("help prob")
    plt.xlabel("probability of public")
    plt.ylim(-0.1, 1.1)
    plt.title('helpCost ='+ str(helpingCost) + ' agentAbility ='+ str(agentAbilityScore) + ' helpReward =' + str(helpingReward))
    plt.show()


    plt.plot(probOfPublicList, boostParamList, color='blue', label = 'boostParameter')
    plt.plot(probOfPublicList, hurtParamList, color='grey', label = 'hurtParameter')
    plt.legend(loc='lower left')
    plt.ylabel("Boost/ Hurt parameter")
    plt.xlabel("Probability of public")
    plt.ylim(-500, 100)
    plt.show()


if __name__ == '__main__':
    main()


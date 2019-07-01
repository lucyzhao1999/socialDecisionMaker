from src.baseDecisionMaker import *
from src.judge import *
from src.wrappers import *
from src.constructedSocialDecisionMaker import *
from visualization.plotResults import *

def main():

# equal merit condition
    merit = [1, 1]
    highBonus = 1000
    lowBonus = 100
    equalBonusList = [0, 100, 500, 1000, 1100]

    highBonusAgentCount = 1
    lowBonusAgentCount = 1
    totalAgentsCount = highBonusAgentCount + lowBonusAgentCount

    createReward = CreateReward(createPartialAllocationList)
    getRewardList = lambda equalBonusList: [createReward(highBonus, lowBonus, equalBonus, highBonusAgentCount, lowBonusAgentCount) for equalBonus in equalBonusList]
    rewardList = getRewardList(equalBonusList)

    lambdaInput = 0.7
    alphaIAList = [np.random.exponential(lambdaInput) for _ in range(10000)]
    baseAgentWeights = (1,)*totalAgentsCount

# first layer
    getUtilityOfInequity = GetUtilityOfInequity(getInequity)
    getBaseDecisionUtility = GetBaseDecisionUtility(getUtilityOfEfficiency, getUtilityOfInequity)

    beta = 0.003
    getActionProbabilityFromUtility = GetActionProbabilityFromUtility(beta)

    getBaseActionProb = GetBaseActionProb(getBaseDecisionUtility, getActionProbabilityFromUtility)
    baseActionProbDictList = getBaseActionProb(rewardList, [baseAgentWeights], alphaIAList, merit)

    baseEqualBonusProb = [baseActionProbDict[baseAgentWeights]['ActionEqual'] for baseActionProbDict in baseActionProbDictList]

# second layer
    alphaPartial = 6
    getAgentsWeightSet = GetAgentsWeightSet(alphaPartial, getAgentsWeight)

    priorOfPartiality = 0.5
    getSingleJudgePartiality = GetSingleJudgePartiality(
        getProbOfAlphaVectorGivenPartial,
        getProbOfAlphaVectorGivenImpartial,
        getJointProbOfPartialAndAlphaGivenAction,
        priorOfPartiality
    )
    getJudgeProb = GetJudgeProb(alphaIAList, getAgentsWeightSet, getBaseActionProb, getSingleJudgePartiality)
    partialAgentNumber = 1
    judgeProbList = getJudgeProb(totalAgentsCount,partialAgentNumber,
                             rewardList,
                             merit)

    partialityUnequalBonus = [judgeProb['Action1'] for judgeProb in judgeProbList]
    partialityEqualBonus = [judgeProb['ActionEqual'] for judgeProb in judgeProbList]

#third layer
    alphaPA = 1350
    getSingleConstructedActionProb = GetSingleConstructedActionProb(alphaPA, getActionProbabilityFromUtility)

    getConstructedActionProb = GetConstructedActionProb(getBaseDecisionUtility, getSingleConstructedActionProb, getActionProbabilityFromUtility, getSingleJudgePartiality,getAgentsWeightSet)

    constructedActionProbList = getConstructedActionProb(totalAgentsCount, partialAgentNumber, rewardList, alphaIAList, merit)

    constructedEqualBonusProb = [constructedActionProb['ActionEqual'] for constructedActionProb in constructedActionProbList]

    barPlotBonusActionProb(equalBonusList, constructedEqualBonusProb, 'Probability of Equal Bonus', 'Equal Merit, Equal Bonus')
    barplotPartiality(equalBonusList, partialityEqualBonus, 'Equal Merit, Equal Bonus')
    barplotPartiality(equalBonusList, partialityUnequalBonus, 'Equal Merit, Unqual Bonus')



if __name__ == '__main__':
    main()


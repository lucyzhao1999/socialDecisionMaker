import numpy as np


class GetBaseActionProb:
    def __init__(self, getBaseDecisionUtility, getActionProbabilityFromUtility):
        self.getBaseDecisionUtility = getBaseDecisionUtility
        self.getActionProbabilityFromUtility = getActionProbabilityFromUtility

    def __call__(self, rewardList, agentWeights, alphaIAList, merit):

        getSingleActionProb = lambda reward, weight, alphaIA: self.getActionProbabilityFromUtility(
            self.getBaseDecisionUtility(reward, [weight, alphaIA], merit))

        meanOfAlphaIACase = lambda actionProbAlphaCaseList: {
            action: np.mean([actionProbAlphaCase[action] for actionProbAlphaCase in actionProbAlphaCaseList]) for action
            in actionList}

        getActionProb = lambda reward, weight, alphaIAList: meanOfAlphaIACase(
            [getSingleActionProb(reward, weight, alphaIA) for alphaIA in alphaIAList])

        actionList = list(rewardList[0].keys())
        allAlphaAICaseProb = [{weight: getActionProb(reward, weight, alphaIAList) for weight in agentWeights} for
                              reward in rewardList]

        return allAlphaAICaseProb


class GetJudgeProb:
    def __init__(self, alphaIAList, getAgentsWeightSet,getBaseActionProb, getSingleJudgePartiality):
        self.alphaIAList = alphaIAList
        self.getAgentsWeightSet = getAgentsWeightSet

        self.getBaseActionProb = getBaseActionProb
        self.getSingleJudgePartiality = getSingleJudgePartiality

    def __call__(self, numberOfAgents,rewardList, merit):

        agentsWeightSet = self.getAgentsWeightSet(numberOfAgents)
        baseActionProbList = self.getBaseActionProb(rewardList, agentsWeightSet, self.alphaIAList, merit)
        judgeProb = [self.getSingleJudgePartiality(agentsWeightSet, baseActionProb) for baseActionProb in baseActionProbList]
        return judgeProb

#
# class GetConstructedActionProb:
#     def __init__(self, getBaseDecisionUtility, getSingleConstructedUtility,
#                  getActionProbabilityFromUtility, getSingleJudgePartiality,
#                  getAgentsWeightSet):
#
#         self.getBaseDecisionUtility = getBaseDecisionUtility
#         self.getSingleConstructedUtility = getSingleConstructedUtility
#         self.getActionProbabilityFromUtility = getActionProbabilityFromUtility
#         self.getSingleJudgePartiality = getSingleJudgePartiality
#         self.getAgentsWeightSet = getAgentsWeightSet
#
#     def __call__(self, numberOfAgents, partialAgentNumber, rewardList, alphaIAList, merit):
#
#         equalWeight = (1,) * numberOfAgents
#         getSingleUtility = lambda reward, alphaIA: self.getBaseDecisionUtility(reward, [equalWeight, alphaIA], merit)
#         baseUtility = [[getSingleUtility(reward, alphaIA) for alphaIA in alphaIAList] for reward in rewardList]
#
#         agentsWeightSet = self.getAgentsWeightSet(numberOfAgents, partialAgentNumber)
#
#         getSingleActionProb = lambda reward, weight, alphaIA: self.getActionProbabilityFromUtility(
#             self.getBaseDecisionUtility(reward, [weight, alphaIA], merit))
#
#         baseActionProbList = [[{weight: getSingleActionProb(reward, weight, alphaIA) for weight in agentsWeightSet}
#                                for alphaIA in alphaIAList]
#                               for reward in rewardList]
#
#         actionList = list(rewardList[0].keys())
#
#
#         partialityList = [[self.getSingleJudgePartiality(agentsWeightSet, alphaBaseActionProbGivenBonus) for
#                          alphaBaseActionProbGivenBonus in equalBonusBaseActionProb] for equalBonusBaseActionProb in
#                         baseActionProbList]
#
#         constructedUtilityGivenAlphaIA = [
#             [self.getSingleConstructedUtility(baseAlphaCase, judgeAlphaCase) for baseAlphaCase, judgeAlphaCase in
#              zip(baseBonusCase, judgeBonusCase)] for baseBonusCase, judgeBonusCase in zip(baseUtility, partialityList)]
#
#         constructedUtilityList = [
#             {action: np.mean([alphaIACase[action] for alphaIACase in equalBonusCase]) for action in actionList} for
#             equalBonusCase in constructedUtilityGivenAlphaIA]
#
#         constructedActionProb = [self.getActionProbabilityFromUtility(constructedUtility) for constructedUtility in constructedUtilityList]
#
#
#         return constructedActionProb
#
#


class GetConstructedActionProb:
    def __init__(self, getBaseDecisionUtility, getSingleConstructedActionProb,
                 getActionProbabilityFromUtility, getSingleJudgePartiality,
                 getAgentsWeightSet):

        self.getBaseDecisionUtility = getBaseDecisionUtility
        self.getSingleConstructedActionProb = getSingleConstructedActionProb
        self.getActionProbabilityFromUtility = getActionProbabilityFromUtility
        self.getSingleJudgePartiality = getSingleJudgePartiality
        self.getAgentsWeightSet = getAgentsWeightSet

    def __call__(self, numberOfAgents, rewardList, alphaIAList, merit):

        equalWeight = (1,) * numberOfAgents
        getSingleUtility = lambda reward, alphaIA: self.getBaseDecisionUtility(reward, [equalWeight, alphaIA], merit)
        baseUtility = [[getSingleUtility(reward, alphaIA) for alphaIA in alphaIAList] for reward in rewardList]

        agentsWeightSet = self.getAgentsWeightSet(numberOfAgents)

        getSingleActionProb = lambda reward, weight, alphaIA: self.getActionProbabilityFromUtility(
            self.getBaseDecisionUtility(reward, [weight, alphaIA], merit))

        baseActionProbList = [[{weight: getSingleActionProb(reward, weight, alphaIA) for weight in agentsWeightSet}
                               for alphaIA in alphaIAList]
                              for reward in rewardList]

        actionList = list(rewardList[0].keys())


        partialityList = [[self.getSingleJudgePartiality(agentsWeightSet, alphaBaseActionProbGivenBonus) for
                         alphaBaseActionProbGivenBonus in equalBonusBaseActionProb] for equalBonusBaseActionProb in
                        baseActionProbList]

        constructedUtilityGivenAlphaIA = [
            [self.getSingleConstructedActionProb(baseAlphaCase, judgeAlphaCase) for baseAlphaCase, judgeAlphaCase in
             zip(baseBonusCase, judgeBonusCase)] for baseBonusCase, judgeBonusCase in zip(baseUtility, partialityList)]

        constructedActionProb = [
            {action: np.mean([alphaIACase[action] for alphaIACase in equalBonusCase]) for action in actionList} for
            equalBonusCase in constructedUtilityGivenAlphaIA]

        # constructedActionProb = [self.getActionProbabilityFromUtility(constructedUtility) for constructedUtility in constructedUtilityList]


        return constructedActionProb











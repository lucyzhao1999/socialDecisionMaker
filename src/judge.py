from itertools import combinations


def getAgentsWeight(alphaPartial, numberOfAgents, partialAgentIndexList):
    value = [-alphaPartial] * numberOfAgents
    for partialAgentIndex in partialAgentIndexList:
        value[partialAgentIndex] = alphaPartial
    return tuple(value)



class GetAgentsWeightSet:
    def __init__(self, alphaPartial, getAgentsWeight):
        self.alphaPartial = alphaPartial
        self.getAgentsWeight = getAgentsWeight

    def __call__(self, numberOfAgents, partialAgentNumber):
        agentsIndex = list(range(numberOfAgents))
        partialAgentIndex = list(combinations(agentsIndex, partialAgentNumber))
        agentsWeightSet = [self.getAgentsWeight(self.alphaPartial, numberOfAgents, partialList) for partialList in partialAgentIndex]
        agentsWeightSet.append((1,) * numberOfAgents)
        return agentsWeightSet



class GetProbOfActionGivenAlpha:
    def __init__(self, getBaseDecisionUtility, getActionProbabilityFromUtility):
        self.getBaseDecisionUtility = getBaseDecisionUtility
        self.getActionProbabilityFromUtility = getActionProbabilityFromUtility

    def __call__(self, reward, agentsWeightSet, alphaIA, merit):
        alphaVectorSet = [[agentWeight, alphaIA] for agentWeight in agentsWeightSet]
        baseUtilityGivenAlpha = {alphaVector: self.getBaseDecisionUtility(reward, alphaVector, merit) for alphaVector in alphaVectorSet}

        alphaVectorSetAndUtilityPair = zip(baseUtilityGivenAlpha.keys(), baseUtilityGivenAlpha.values())
        baseActionProbGivenAlpha = {alphaVector[0]: self.getActionProbabilityFromUtility(baseUtility) for alphaVector, baseUtility in alphaVectorSetAndUtilityPair}
        return baseActionProbGivenAlpha


def getProbOfAlphaVectorGivenPartial(agentsWeightSet):
    partialWeights = agentsWeightSet[:-1]
    probOfAlphaGivenPartial = {agentsWeight: 1.0 / len(partialWeights) for
                               agentsWeight in agentsWeightSet}
    numberOfAgents = len(agentsWeightSet[0])
    impartialAgentsWeight = (1,) * numberOfAgents
    probOfAlphaGivenPartial[impartialAgentsWeight] = 0
    return probOfAlphaGivenPartial

def getProbOfAlphaVectorGivenImpartial(agentsWeightSet):
    numberOfAgents = len(agentsWeightSet[0])
    probOfAlphaGivenImpartial = {agentsWeight: 0 for agentsWeight in agentsWeightSet}
    impartialAgentsWeight = (1,) * numberOfAgents
    probOfAlphaGivenImpartial[impartialAgentsWeight] = 1
    return probOfAlphaGivenImpartial


def getJointProbOfPartialAndAlphaGivenAction(baseActionProbGivenAlpha, probOfAlphaGivenPartial, priorOfPartiality):

    actionList = list((baseActionProbGivenAlpha.values()))[0].keys()
    agentsWeightSet = baseActionProbGivenAlpha.keys()

    prob = {action: {agentWeight: baseActionProbGivenAlpha[agentWeight][action] *
                                  probOfAlphaGivenPartial[agentWeight] *
                                  priorOfPartiality
                     for agentWeight in agentsWeightSet}
            for action in actionList}
    return prob

class GetSingleJudgePartiality:
    def __init__(self,
                 getProbOfAlphaVectorGivenPartial,
                 getProbOfAlphaVectorGivenImpartial,
                 getJointProbOfPartialAndAlphaGivenAction,
                 priorOfPartiality
                 ):
        self.getProbOfAlphaVectorGivenPartial = getProbOfAlphaVectorGivenPartial
        self.getProbOfAlphaVectorGivenImpartial = getProbOfAlphaVectorGivenImpartial
        self.getJointProbOfPartialAndAlphaGivenAction = getJointProbOfPartialAndAlphaGivenAction
        self.priorOfPartiality = priorOfPartiality
        self.priorOfImpartiality = 1-priorOfPartiality

    def __call__(self, agentsWeightSet, baseActionProbGivenAlpha):

        # baseActionProbGivenAlpha = self.getProbOfActionGivenAlpha(reward, agentsWeightSet, alphaIA, merit)

        probOfAlphaGivenPartial = self.getProbOfAlphaVectorGivenPartial(agentsWeightSet)
        jointPartialProbGivenAction = getJointProbOfPartialAndAlphaGivenAction(baseActionProbGivenAlpha, probOfAlphaGivenPartial, self.priorOfPartiality)

        probOfAlphaGivenImpartial = self.getProbOfAlphaVectorGivenImpartial(agentsWeightSet)
        jointImpartialProbGivenAction = getJointProbOfPartialAndAlphaGivenAction(baseActionProbGivenAlpha, probOfAlphaGivenImpartial, self.priorOfImpartiality)

        marginalizeWeight = lambda jointProb: {action: sum(jointProb[action].values()) for action in jointProb.keys()}
        marginalizedProbOfPartialGivenAction = marginalizeWeight(jointPartialProbGivenAction)
        marginalizedProbOfImpartialGivenAction = marginalizeWeight(jointImpartialProbGivenAction)

        normalizeProb = lambda partialProb, impartialProb: partialProb / (partialProb + impartialProb)
        actionList = marginalizedProbOfImpartialGivenAction.keys()
        normalizedProbOfPartialGivenAction = {action: normalizeProb(marginalizedProbOfPartialGivenAction[action], marginalizedProbOfImpartialGivenAction[action]) for action in actionList}

        return normalizedProbOfPartialGivenAction


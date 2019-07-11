import matplotlib.pyplot as plt
import os

def plotEqualBonusPartiality(equalBonus, partialProbForUnequalBonus, partialProbForEqualBonus):
    plt.plot(equalBonus, partialProbForUnequalBonus, color='grey', label='Unequal Allocation')
    plt.plot(equalBonus, partialProbForEqualBonus, color='blue', label='Equal Allocation')
    plt.legend(loc='upper left')
    plt.ylabel("Partiality")
    plt.xlabel("Equal Bonus")
    plt.ylim(-0.1, 1.1)
    plt.savefig(os.path.join('equalBonusPartiality'))
    plt.show()


def plotEqualBonusActionProb(equalBonusList, actionProbForBaseDM, actionProbForConstructed):
    plt.plot(equalBonusList, actionProbForBaseDM, color='blue', label='Base')
    plt.plot(equalBonusList, actionProbForConstructed, color='grey', label='Constructed')
    plt.legend(loc='lower right')
    plt.ylabel("Probability of Equal Bonus")
    plt.xlabel("Equal Bonus")
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join('equalBonusActionProbComparison'))
    plt.show()


def barPlotBonusActionProb(equalBonusList, actionProb, yLabel, plotTitle):
    plt.bar([str(bonus) for bonus in equalBonusList], actionProb, color='blue', edgecolor='black')
    plt.ylabel(yLabel)
    plt.xlabel("Equal Bonus")
    plt.title(plotTitle)
    plt.ylim((0, 1))
    plt.savefig(os.path.join(plotTitle))
    plt.show()


def barplotPartiality(equalBonusList, partiality, plotTitle):
    plt.bar([str(bonus) for bonus in equalBonusList], partiality, color='blue', edgecolor='black')
    plt.ylabel("Partiality")
    plt.xlabel("Equal Bonus")
    plt.title(plotTitle)
    plt.ylim((0, 1))
    plt.savefig(os.path.join(plotTitle))
    plt.show()

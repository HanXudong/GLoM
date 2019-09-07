from scipy import spatial
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import numpy as np

def Delta_E(Color1, Color2):
    color1_rgb = sRGBColor(Color1[0], Color1[1], Color1[2])
    color2_rgb = sRGBColor(Color2[0], Color2[1], Color2[2])
    # Convert from RGB to Lab Color Space
    color1_lab = convert_color(color1_rgb, LabColor)
    # Convert from RGB to Lab Color Space
    color2_lab = convert_color(color2_rgb, LabColor)
    return delta_e_cie2000(color1_lab, color2_lab)


def evaluation(Train_truples, Test_truples, Pred, E_Rmu, E_Tmu):

    Choosen_instances_reference = set([i[0] for i in Train_truples])
    Choosen_instances_modifier = set([i[1] for i in Train_truples])
    Choosen_instances_target = set([i[2] for i in Train_truples])

    CosineD = [1 - spatial.distance.cosine(Pred[i]-E_Rmu[i], E_Tmu[i]-E_Rmu[i]) for i in range(len(E_Rmu))]
    # DeltaE = [Delta_E(Pred[i]-E_Rmu[i], E_Tmu[i]-E_Rmu[i]) for i in range(len(E_Rmu))]
    DeltaE = [Delta_E(Pred[i], E_Tmu[i]) for i in range(len(E_Rmu))]

    Seen_pairings = []
    Unseen_Pairings = []
    Unseen_Reference = []
    Unseen_Modifier = []
    Fully_Unseen = []
    DE_Seen_pairings = []
    DE_Unseen_Pairings = []
    DE_Unseen_Reference = []
    DE_Unseen_Modifier = []
    DE_Fully_Unseen = []


    for i in range(len(Test_truples)):
        r, m, t = Test_truples[i]
        s_unseen_r = not r in Choosen_instances_reference
        s_unseen_m = not m in Choosen_instances_modifier
        if s_unseen_r and s_unseen_m:
            Fully_Unseen.append(CosineD[i])
            DE_Fully_Unseen.append(DeltaE[i])
        elif s_unseen_r:
            Unseen_Reference.append(CosineD[i])
            DE_Unseen_Reference.append(DeltaE[i])
        elif s_unseen_m:
            Unseen_Modifier.append(CosineD[i])
            DE_Unseen_Modifier.append(DeltaE[i])
        else:
            if Test_truples[i] in Train_truples:
                Seen_pairings.append(CosineD[i])
                DE_Seen_pairings.append(DeltaE[i])
            else:
                Unseen_Pairings.append(CosineD[i])
                DE_Unseen_Pairings.append(DeltaE[i])


    CosinSimilarity = {}
    CosinSimilarity["Unseen_Pairings"] = np.mean(Unseen_Pairings)
    CosinSimilarity["Unseen_Reference"] = np.mean(Unseen_Reference) 
    CosinSimilarity["Unseen_Modifier"] = np.mean(Unseen_Modifier)
    CosinSimilarity["Fully_Unseen"] = np.mean(Fully_Unseen)
    CosinSimilarity["Seen_pairings"] = np.mean(Seen_pairings)
    CosinSimilarity["Overall"] =  np.mean(CosineD)

    Delt = {}
    Delt["Unseen_Pairings"] = np.mean(DE_Unseen_Pairings)
    Delt["Unseen_Reference"] = np.mean(DE_Unseen_Reference) 
    Delt["Unseen_Modifier"] = np.mean(DE_Unseen_Modifier)
    Delt["Fully_Unseen"] = np.mean(DE_Fully_Unseen)
    Delt["Seen_pairings"] = np.mean(DE_Seen_pairings)
    Delt["Overall"] = np.mean(DeltaE)

    return CosinSimilarity, Delt

def Test_conditios(Train_truples, Test_truples):

    Choosen_instances_reference = set([i[0] for i in Train_truples])
    Choosen_instances_modifier = set([i[1] for i in Train_truples])
    Choosen_instances_target = set([i[2] for i in Train_truples])

    Seen_pairings = []
    Unseen_Pairings = []
    Unseen_Reference = []
    Unseen_Modifier = []
    Fully_Unseen = []


    for i in range(len(Test_truples)):
        r, m, t = Test_truples[i]
        s_unseen_r = not r in Choosen_instances_reference
        s_unseen_m = not m in Choosen_instances_modifier
        if s_unseen_r and s_unseen_m:
            Fully_Unseen.append(i)
        elif s_unseen_r:
            Unseen_Reference.append(i)
        elif s_unseen_m:
            Unseen_Modifier.append(i)
        else:
            if Test_truples[i] in Train_truples:
                Seen_pairings.append(i)
            else:
                Unseen_Pairings.append(i)

    Conditions = {}
    Conditions["Fully_Unseen"] = Fully_Unseen
    Conditions["Unseen_Reference"] = Unseen_Reference
    Conditions["Unseen_Modifier"] = Unseen_Modifier
    Conditions["Seen_pairings"] = Seen_pairings
    Conditions["Unseen_Pairings"] = Unseen_Pairings

    return Conditions
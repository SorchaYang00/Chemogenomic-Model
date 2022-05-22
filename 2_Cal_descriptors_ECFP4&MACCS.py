import openbabel as ob
from rdkit.Chem import AllChem as Chem
import numpy as np
from rdkit.Chem import MACCSkeys

#############################################

LOAD_FILE = r"E:\mols_for_prediction.csv"
SAVE_FILE = r"E:\mols_ECFP4.csv"
# SAVE_FILE = r"E:\mols_MACCS.csv"

def obsmitosmile(smi):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = ob.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile
    

def getMol(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return mol
    else:
        return Chem.MolFromSmiles(obsmitosmile(smi))


def main(smis):
    mols = [getMol(smi) for smi in smis]
    fps = []
    for mol in mols:
        vector = Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)   #####   ECFP4   #########
        # vector = MACCSkeys.GenMACCSKeys(mol)      #####   MACCS   #########
        feature = vector.ToBitString()
        fps.append(feature)
    fps = np.array(fps).astype(np.int8)
    return fps



if '__main__'==__name__:
    # data = pd.read_csv(LOAD_FILE)
    # smis=data['SMILE']

    smis = ['c1ccccc1']*10
    fps = main(smis)
    print(fps)
#    fps.to_csv(SAVE_FILE, index=False)
 

    

   

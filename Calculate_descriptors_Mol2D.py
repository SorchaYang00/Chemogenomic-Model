import pandas as pd
import openbabel as ob
from rdkit.Chem import AllChem as Chem
from multiprocessing import Pool
from PyBioMed.PyMolecule import constitution, topology, connectivity, kappa
from PyBioMed.PyMolecule import basak, charge, moe


############################################
LOAD_FILE = r"E:\mols_for_prediction.csv"
SAVE_FILE = r"E:\mols_Mol2d.csv"

N_JOBS = 28


###############################################
class PbmFeatureVector(object):
    """
    """
    def __init__(self, mols, n_jobs=1):
        self.mols = mols
        self.n_jobs = n_jobs if n_jobs >= 1 else None
        
    def Getconstitution(self):
        ps = Pool(self.n_jobs)
        fps = ps.map_async(constitution.GetConstitutional, self.mols).get()
        ps.close()
        ps.join()
        return pd.DataFrame(fps)

    def Gettopology(self):
        ps = Pool(self.n_jobs)
        fps = ps.map_async(topology.GetTopology, self.mols).get()
        ps.close()
        ps.join()
        return pd.DataFrame(fps)
    
    def Getconnectivity(self):
        ps = Pool(self.n_jobs)
        fps = ps.map_async(connectivity.GetConnectivity, self.mols).get()
        ps.close()
        ps.join()
        return pd.DataFrame(fps)

    def Getkappa(self):
        ps = Pool(self.n_jobs)
        fps = ps.map_async(kappa.GetKappa, self.mols).get()
        ps.close()
        ps.join()
        return pd.DataFrame(fps)

    def Getbasak(self):
        ps = Pool(self.n_jobs)
        fps = ps.map_async(basak.Getbasak, self.mols).get()
        ps.close()
        ps.join()
        return pd.DataFrame(fps)
    
    def Getcharge(self):
        ps = Pool(self.n_jobs)
        fps = ps.map_async(charge.GetCharge, self.mols).get()
        ps.close()
        ps.join()
        return pd.DataFrame(fps)
    
    def Getmoe(self):
        ps = Pool(self.n_jobs)
        fps = ps.map_async(moe.GetMOE, self.mols).get()
        ps.close()
        ps.join()
        return pd.DataFrame(fps)
    
    def GetAllFeatures(self):
        funl = [
                'self.Getconstitution()',
                'self.Gettopology()',
                'self.Getconnectivity()',
                'self.Getkappa()',
                'self.Getbasak()',
                'self.Getcharge()',
                'self.Getmoe()',
            ]
        
        for func in funl:
            yield eval(func)
            print(func)
  
  
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

def main(smis, n_jobs=28):
    mols = [getMol(smi) for smi in smis]
    vector = PbmFeatureVector(mols, n_jobs)
    features = pd.concat(vector.GetAllFeatures(),axis=1)
    
    return features



if '__main__'==__name__:
    # data = pd.read_csv(LOAD_FILE)
    # smis=data['SMILE']

    smis = ['c1ccccc1']*10
    features = main(smis, N_JOBS)
    print(features)
#    features.to_csv(SAVE_FILE, index=False)
  

import os
import re
import sys
import time
import shutil
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

########## Executable paths ##########

# For exaLearn systems
OBABEL_PATH = "/usr/bin/obabel"
ADT_PATH = "/clusterfs/csdata/pkg/autodock-gpu-153/AutoDock-GPU/bin/autodock_gpu_64wi"
# For Summit systems
#OBABEL_PATH = "/gpfs/alpine/syb105/proj-shared/Personal/manesh/BIN/openbabel/summit/build/bin/obabel"
#ADT_PATH = "/gpfs/alpine/syb105/proj-shared/Personal/gabrielgaz/Apps/summit/autoDockGPU2/bin/autodock_gpu_64wi"

########### Receptor Files ###########

# NSP15, site A3H
RECEPTOR_FILE = "NSP15_6W01_A_3_H_receptor.pdbqt"

######################################

def get_dock_score(states, args=None):

    #Debugging flag
    DEBUG=False
    #Tmp files to save 
    #  0 for none, 1 for last round, 2 for all
    TMP_SAVE=1
    if (TMP_SAVE==2): print("WARNING: You are set to save all AutoDock-GPU temporary files, for large runs this may be excessive.  To change please alter TMP_SAVE in src/reward/adtgpu/get_score.py.")

    is_list = 1
    if not isinstance(states, list):
        states = [states]
        is_list = 0
    #Check for any failed translations
    num_nones = states.count(None)
    if num_nones > 0:
        print("\nWARNING: {} case(s) of NoneType in mol list, signals failure in MolFromSmiles translation\n".format(num_nones))
    if(DEBUG): print("Number of smiles to score: {}".format(len(states)))
    #if(DEBUG): print("List of mols:\n{}".format('\n'.join(states)))

    #Setup paths
    if(args and args.obabel_path!=''): obabel_path=args.obabel_path
    else: obabel_path=OBABEL_PATH
    if(args and args.adt_path!=''): adt_path=args.adt_path
    else: adt_path=ADT_PATH
    if(args and args.receptor_file!=''): receptor_file="./src/reward/adtgpu/receptor/"+args.receptor_file
    else: receptor_file="./src/reward/adtgpu/receptor/"+RECEPTOR_FILE
    if(args and args.run_id!=''): run_dir="./src/reward/adtgpu/autodockgpu"+str(args.run_id)
    else: run_dir="./src/reward/adtgpu/autodockgpu"
    if(DEBUG): print("adttmp: {}".format(run_dir))

    #Check that input file path exist
    if not os.path.exists(receptor_file):
        exit("Receptor file does not exist: {}".format(receptor_file))   
    #Create output dirs
    ligands_dir="/ligands"
    if not os.path.exists(run_dir): 
        os.makedirs(run_dir)
    if not os.path.exists(run_dir+ligands_dir):
        os.makedirs(run_dir+ligands_dir)

    #Loop over mols to convert to pdbqt
    ligs_list=[]
    sm_counter=1
    for mol in states:
        VALID=True
        if (mol==None): VALID=False; print("Processing: None -- skipping checks")
        
        #Step 1 - Filtering
        if(VALID):    
            if(DEBUG and mol!=None): print("Processing: {}".format(Chem.MolToSmiles(mol)))
            try: 
                #Prepare SMILES for conversion, convert to pdb
                mol_with_H=Chem.AddHs(mol)
                ret = AllChem.EmbedMolecule(mol_with_H)
                if (ret == -1): raise Exception(f'Exception: AllChem.EmbedMolecule returned {ret}')
                ret = AllChem.MMFFOptimizeMolecule(mol_with_H)
                if (ret == -1): raise Exception(f'Exception: AllChem.MMFFOptimizeMolecule returned {ret}')
                if(DEBUG): print("Printing MolToPDBBlock:\n".format(Chem.MolToPDBBlock(mol_with_H)))
            except Exception as e:
                print("SMILES error on filtering: {}".format(Chem.MolToSmiles(mol_with_H)))
                print(e)
                VALID=False   
 
        #Step 2 - pdb -> pdbqt with obabel
        if(VALID): 
            #Create temp directory needed for obabel
            tmp_file=run_dir+ligands_dir+"/ligand"+str(sm_counter)+".pdb"
            with open(tmp_file,'w') as f:
                f.write(Chem.MolToPDBBlock(mol_with_H))

            #Create name for output pdbqt file
            ligand_out=run_dir+ligands_dir+"/ligand"+str(sm_counter)+".pdbqt"

            #Convert pdb to pdbqt
            cmd=obabel_path+" --partialcharge gasteiger --addfilename -ipdb "
            cmd+=tmp_file+" -opdbqt -O "+ligand_out
            if(DEBUG): print("\nCmd to run:\n{}".format(cmd))
            if(DEBUG): subprocess.Popen(cmd,shell=True).wait()
            else: subprocess.Popen(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).wait()
            if(DEBUG): print("Done!")

            #Add ligand to ligs_list
            #os.remove(tmp_file)
            ligand_store_file=ligand_out.split('/')[-1][:-6]
            ligs_list.append(ligand_store_file)
        else: #invalid SMILES
            ligs_list.append(None)
        sm_counter+=1
    if(DEBUG): print("ligs_list:\n{}".format(ligs_list))

    #Step 3 - AutoDock-GPU
    #Setup (create list list files) and run AutoDock-GPU
    pred_dock_score=[]
    if(len(ligs_list)>0 and not all(x==None for x in ligs_list)):
        #Get stub name of receptor and field file
        receptor_dir='/'.join(receptor_file.split('/')[:-1])
        receptor_stub=receptor_file.split('/')[-1][:-6] #rm .pdbqt=6
        if(DEBUG): print("\nReceptor dir:  {}".format(receptor_dir))
        if(DEBUG): print("Receptor stub: {}".format(receptor_stub))
        receptor_field=receptor_stub+".maps.fld"

        #Create run file for Autodock-gpu
        run_file=run_dir+"/ligs_list.runfile"
        run_file_lbl="ligs_list.runfile"
        with open(run_file,'w') as f:
            f.write(receptor_field+'\n')
            for lig in ligs_list:
                if lig != None:
                    f.write("ligands/"+lig+".pdbqt\n")
                    f.write("ligands/"+lig+'\n')

        #Copy map files to run dir
        cmd="cp "+receptor_dir+"/"+receptor_stub+"* "+run_dir
        if(DEBUG): print("\nCopy cmd to run: {}".format(cmd))
        if(DEBUG): subprocess.Popen(cmd,shell=True).wait()
        else: subprocess.Popen(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).wait()

        #Set up autodock-gpu run command
        cmd=adt_path + " -filelist "+run_file_lbl+" -nrun 10"
        if(DEBUG): print("\nAutodock cmd to run: {}".format(cmd))

        #Run autodock-gpu (in run_dir and move back)
        cur_dir=os.getcwd()
        os.chdir(run_dir)
        if(DEBUG): subprocess.Popen(cmd,shell=True).wait()
        else: subprocess.Popen(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).wait()
        os.chdir(cur_dir)

        #Read final scores into list
        for lig in ligs_list:
            if lig != None:
                #Parse for final score
                lig_path=run_dir+ligands_dir+"/"+lig+".dlg"
                if not os.path.exists(lig_path):
                    print("ERROR: No such file {}\nDocking score marked as 0.00".format(lig_path))
                    pred_dock_score.append(0.00)
                else: 
                    grep_cmd = "grep -2 \"^Rank \" "+lig_path+" | head -5 | tail -1 | cut -d \'|\' -f2 | sed \'s/ //g\'"
                    grep_out=os.popen(grep_cmd).read()
                    pred_dock_score.append(float(grep_out.strip()))
            else:#invalid SMILES
                print("WARNING: lig=None.  Docking score marked as 0.00")
                pred_dock_score.append(0.00)
    else:#ligs list is empty
        print("WARNING: ligs_list is empty or all None, zeroing all scores...")
        for s in range(0,sm_counter-1):
            pred_dock_score.append(0.00)

    #Remove or move temporary files based on TMP_SAVE
    if (TMP_SAVE==1): 
        shutil.rmtree(run_dir+"_tmpfiles_lastround", ignore_errors=True)
        shutil.move(run_dir, run_dir+"_tmpfiles_lastround") 
    elif (TMP_SAVE==2): shutil.move(run_dir, run_dir+"_tmpfiles/"+time.strftime("%Y%m%d-%H%M%S"))
    else: shutil.rmtree(run_dir, ignore_errors=True)

    if(DEBUG): print("Docking Scores: {}".format(pred_dock_score))
    if not is_list:
        pred_dock_score = pred_dock_score[0]
    return pred_dock_score

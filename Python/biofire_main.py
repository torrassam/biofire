# -*- coding: utf-8 -*-
"""
Code for Master Thesis "How fires shape biodiversity in plant communities: a study using a stochastic dynamical model" (Torrassa, 2023)
__date__ = '20240618'
__version__ = '2.0.1'
__author__ =
    'Matilde Torrassa' (matilde.torrassa@cimafoundation.org')

General command line:
python biofire_main.py -settings_file "biofire_setting.json"

Version(s):
20230405 (1.0.0)
20240221 (1.1.0)
20240222 (1.1.1)
20240305 (2.0.0)
20240618 (2.0.1) -> fifth version: new generationn functions for C and M based on linear regression of the Med Biome species
"""

import os
import json
import logging
import glob
from time import time
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

import biofire_tools as bft

# Script Main
def main():
    rng0 = default_rng()

    # Get algorithm settings
    [file_script, file_settings] = get_args()

    fjson = open(file_settings)
    data_settings = json.load(fjson)
    bft.settings(file_settings) # send setting arguments to biofire_tools.py

    # Set algorithm logging
    os.makedirs(data_settings['log']['folder'], exist_ok=True)
    set_logging(logger_file=os.path.join(data_settings['log']['folder'], data_settings['log']['filename']))

    # -------------------------------------------------------------------------------------

    # Info algorithm
    logging.info('Generation and Simulation of N-species ecosystems and invasive experiments')
    logging.info(f'Experiment type: {data_settings["algorithm"]["general"]["experiment"]}')
    logging.info(f'Number of ecosystems: {data_settings["algorithm"]["generation"]["n_communities"]}')
    logging.info(f'--> Number of initial species: {data_settings["algorithm"]["generation"]["n_species"]}')
    
    # -------------------------------------------------------------------------------------

    # Time algorithm information
    start_time = time()

    # geretate the community with the parameters set
    noise = data_settings["algorithm"]["generation"]["noise"]
    NP = data_settings["algorithm"]["generation"]["n_species"]
    Ncoms = data_settings["algorithm"]["generation"]["n_communities"]
    
    nt = data_settings["algorithm"]["simulation"]["repetition"]

    # -------------------------------------------------------------------------------------

    # PARAMETERS from json and function file

    exp_sname = data_settings["algorithm"]["general"]["experiment"][0:3] #experiment short-name for the dir and files names ("com_" or "inv_")

    bft.initial_conditions(NP)

    NN = bft.NN
    N0 = bft.N0
    N1 = bft.N1
    B0ic = np.array(bft.B0ic)

    f_path = data_settings["output"]["data"]["path"].format(exp_sname, NP)
    os.makedirs(f_path, exist_ok=True)

    fig_path = data_settings["output"]["figure"]["path"].format(exp_sname, NP)
    os.makedirs(fig_path, exist_ok=True)

    if data_settings["algorithm"]["general"]["figure"]:
        mytab = bft.set_color_tab(NP)

    # -------------------------------------------------------------------------------------

    # GENERATION of the experiment ecosystem - if "generation" is true

    if data_settings["algorithm"]["general"]["generation"]:

        logging.info('--> Generation ... START!')
        logging.info(f'--> Generation distribution: {data_settings["algorithm"]["generation"]["distribution"]} with {data_settings["algorithm"]["generation"]["noise"]} noise')

        # check on the pre-existing communities generated with same parameters
        f_list = glob.glob(os.path.join(f_path, '*info.dat'))
        nc_old = len(f_list)

        for nc in range(Ncoms-nc_old):
            
            # Experiment selection from the json file: "community", "invasive"

            if data_settings["algorithm"]["general"]["experiment"]=="community":

                # Community Generation distribution from the json file: "RU", "RCL", "RCE"

                if data_settings["algorithm"]["generation"]["distribution"]=="RU":
                    bft.rand_uniform_community(rng=rng0, nnew=NP)

                elif data_settings["algorithm"]["generation"]["distribution"]=="RCL":
                    bft.rand_linear_community(rng=rng0, nnew=NP, noise=noise)

                # NEW! generation with meditteranean traits
                elif data_settings["algorithm"]["generation"]["distribution"]=="MED":
                    bft.rand_linear_community_med(rng=rng0, nnew=NP)

                elif data_settings["algorithm"]["generation"]["distribution"]=="RCE":
                    bft.rand_exponential_community(rng=rng0, nnew=NP, noise=noise)

                else:
                    logging.error("Distribution name not valid")
                    return

            elif data_settings["algorithm"]["general"]["experiment"]=="invasive":
                
                bft.med_community()

                # Community Generation distribution from the json file: "RU", "RCL", "RCE"
                
                if data_settings["algorithm"]["generation"]["distribution"]=="RU":
                    bft.ru_invasive(rng=rng0, nnew=1)
                
                elif data_settings["algorithm"]["generation"]["distribution"]=="RCL":
                    bft.rcl_invasive(rng=rng0, nnew=1, cvar=noise)
                
                elif data_settings["algorithm"]["generation"]["distribution"]=="RCE":
                    bft.rce_invasive(rng=rng0, nnew=1, cvar=noise)
                
                else:
                    logging.error("Distribution name not valid")
                    return

            else:
                logging.error("Experiment name not valid")
                return

            f_info = os.path.join(f_path, data_settings["output"]["data"]["file_info"].format(exp_sname, data_settings["algorithm"]["generation"]["distribution"], noise, NP, nc+nc_old))

            bft.eco_info_file(f_info)
        
        logging.info('--> Generation ... DONE!')

    # -------------------------------------------------------------------------------------

    # SIMULATION of the experiment ecosystem - if "simulation" is true

    if data_settings["algorithm"]["general"]["simulation"]:

        logging.info('--> Simulation ... START!')
        logging.info(f'Simulation period: {data_settings["algorithm"]["simulation"]["runtime"]} years')
        logging.info(f'Integration time-step: {data_settings["algorithm"]["simulation"]["delta_day"]} days')
        logging.info(f'Number of simulation repetition period: {data_settings["algorithm"]["simulation"]["repetition"]}')

        # select the communities to simulate if specified in the json file, otherwise consider all those generated so far
        if len(data_settings["algorithm"]["simulation"]["n_coms"]):
            # check that the required communities do exist
            if max(data_settings["algorithm"]["simulation"]["n_coms"]) > len(glob.glob(os.path.join(f_path, f'*-info.dat')))-1:
                logging.error(f"Can't find the following communities:{data_settings["algorithm"]["simulation"]["n_coms"]}")
                return
            else:
                f_list = [glob.glob(os.path.join(f_path, f'*-{i}-info.dat'))[0] for i in data_settings["algorithm"]["simulation"]["n_coms"]]
        else:
            f_list = glob.glob(os.path.join(f_path, f'*-info.dat'))
        NC = len(f_list)

        # select the initial condition to plot if specified in the json file - only when plotting the simulations
        if data_settings["algorithm"]["general"]["figure"] and len(data_settings["algorithm"]["simulation"]["n_init"]):
            if max(data_settings["algorithm"]["simulation"]["n_init"]) > NP+2:
                logging.error(f"Initial conditions out of range: {data_settings["algorithm"]["simulation"]["n_init"]}")
                return
            else:
                Ncond = data_settings["algorithm"]["simulation"]["n_init"]
        else:
            Ncond = range(int(np.size(B0ic)/NP))  # number of initial conditions

        progress_bar = tqdm(total=(NC))

        for f_info in f_list:
            
            # variables initialization
            bft.initial_conditions(NP)
            b0 = np.zeros(NP)
            fave = 0.0  # fire return time average
            bout_nat = np.zeros([NP,int(N1)])
            bout = np.zeros([NP,int(NN-N0)])
            bave_tot = []
            fave_tot = []

            nc = os.path.basename(f_info).split('-')[3]

            bft.set_traits(f_info)

            bft.eco_info()
            iINV = np.nonzero(bft.spp)[0]
            
            it = 0
            while it<nt:
                
                initcond = 0
                for initcond in Ncond:

                    logging.info(f'--> Simulation ... Initial condition {initcond+1}/{len(Ncond)}')
                    
                    firevf = bft.firevf
            
                    # variables initialization
                    b0 = np.copy(B0ic[initcond,:])
                    logging.info(f'Space occupation at t= 0 yr : {b0*(b0>1.0e-10)}')

                    # create the figure canva if "figure"=true
                    if data_settings["algorithm"]["general"]["figure"]:
                        fig, ax = plt.subplots(figsize=(6,3), dpi=100)
                        ax.set(xlim=(0, NN/365), ylim=(0, 1))        

                    if data_settings["algorithm"]["general"]["experiment"]=="invasive":
                        # DYNAMIC with native species only for the initial time (initT, N1)
                        bb, fv = bft.dyn(b0, bout_nat, N1, firevf)
                        b0 = bb[:,-1]
                        logging.info(f'Space occupation at t= {N1/365} yr : {b0*(b0>1.0e-10)}')        
                        if len(fv)>0:
                            firevf = fv[-1]
                        # Alien species initial condition: 0.01
                        b0[iINV] = 0.01

                        # if data_settings["algorithm"]["general"]["figure"]:
                        #     xx = np.arange(0, N1)/365
                        #     for i in range(NP):
                        #         ax.plot(xx, bb[i], label=f'pft {i}', color=mytab[i])

                    # DYNAMIC for the first half of the simulation (maxT/2, N0)
                    bb, fv = bft.dyn(b0, bout, N0, firevf)
                    logging.info('--> Simulation ... 50%')
                    b0 = bb[:,-1]
                    logging.info(f'Space occupation at t= {N0/365} yr : {b0*(b0>1.0e-10)}')
                    if len(fv)>0:
                        firevf = fv[-1]
                    
                    # PLOTTING THE CURRENT SIMULATION

                    if data_settings["algorithm"]["general"]["figure"]:
                        xx = np.arange(0, N0)/365
                        for i in range(NP):
                            ax.plot(xx, bb[i], label=f'pft {i}', color=mytab[i])

                    # DYNAMIC for the second half of the simulation (maxT/2, N0)
                    bb, fv = bft.dyn(b0, bout, N0, firevf)
                    b0 = bb[:,-1]
                    logging.info('--> Simulation ... 100%')
                    logging.info(f'Space occupation at t= {NN/365} yr : {b0*(b0>1.0e-10)}')

                    # -------------------------------------------------------------------------------------

                    # AVERAGE FIRE RETURN PERIOD

                    nf = len(fv)
                    fvec = np.array(fv)
                    if nf > 0:
                        fave = np.mean(fvec)
                        fmed = np.median(fvec)
                        # print('\naverage fire return time ', fave)
                        fave_tot.append([fave, fmed, nf])
                    
                    bave = np.mean(bb, axis=1)
                    bave = (bave*(bave > 1.0e-10)).tolist()
                    bave_tot.append(bave)
                    # print('average occupied space ', bave,'\n')

                    # -------------------------------------------------------------------------------------

                    # PLOTTING THE CURRENT SIMULATION

                    if data_settings["algorithm"]["general"]["figure"]:

                        logging.info('--> Plotting the current simulation ... START')

                        xx = np.arange(N0, NN)/365
                        for i in range(NP):
                            ax.plot(xx, bb[i], color=mytab[i])
                        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
                        plt.xlabel('Time (yr)')
                        plt.ylabel('Plant cover (b)')

                        fig.savefig(os.path.join(fig_path, data_settings["output"]["figure"]["fig_name"].format(exp_sname, data_settings["algorithm"]["generation"]["distribution"], noise, NP, nc, int(NN/365), initcond)), dpi=100, bbox_inches='tight')

                        logging.info('--> Plotting the current simulation ... DONE!')
                    
                    # continue with the simulation of the next initial condition
                    initcond += 1
                # continue with simulation repetition of the same community
                it += 1
            
            # -------------------------------------------------------------------------------------

            # WRITING IN FILES

            if data_settings["algorithm"]["general"]["save_files"]:

                logging.info('--> Writing in files ... START')

                f_bave = os.path.join(f_path, data_settings["output"]["data"]["file_bave"].format(exp_sname, data_settings["algorithm"]["generation"]["distribution"], noise, NP, nc, int(NN/365)))
                f_fave = os.path.join(f_path, data_settings["output"]["data"]["file_fave"].format(exp_sname, data_settings["algorithm"]["generation"]["distribution"], noise, NP, nc, int(NN/365)))

                with open(f_bave,'w') as file_bave:
                    for bave in bave_tot:
                        for bv in bave:
                            file_bave.write("{}\t".format(bv))
                        file_bave.write('\n')
                
                with open(f_fave,'w') as file_fave:
                    for fave in fave_tot:
                        for fv in fave:
                            file_fave.write("{}\t".format(fv))
                        file_fave.write('\n')

                logging.info('--> Writing in files ... DONE!')

            progress_bar.update()

            # continue with the simulation of the next community

        logging.info('--> Simulation ... DONE!')


# -------------------------------------------------------------------------------------
        
# Method to set logging information
def set_logging(logger_file='log.txt', logger_format=None):
    if logger_format is None:
        logger_format = '%(asctime)s %(name)-12s %(levelname)-8s ' \
                        '%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s'

    # Remove old logging file
    if os.path.exists(logger_file):
        os.remove(logger_file)

    # Set level of root debugger
    logging.root.setLevel(logging.INFO)

    # Open logging basic configuration
    logging.basicConfig(level=logging.INFO, format=logger_format, filename=logger_file, filemode='w')

    # Set logger handle
    logger_handle_1 = logging.FileHandler(logger_file, 'w')
    logger_handle_2 = logging.StreamHandler()
    # Set logger level
    logger_handle_1.setLevel(logging.INFO)
    logger_handle_2.setLevel(logging.INFO)
    # Set logger formatter
    logger_formatter = logging.Formatter(logger_format)
    logger_handle_1.setFormatter(logger_formatter)
    logger_handle_2.setFormatter(logger_formatter)
    # Add handle to logging
    logging.getLogger('').addHandler(logger_handle_1)
    logging.getLogger('').addHandler(logger_handle_2)

# -------------------------------------------------------------------------------------

# Method to get script argument(s)
def get_args():

    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-time_start', action="store", dest="alg_time_start")
    parser_handle.add_argument('-time_end', action="store", dest="alg_time_end")
    parser_values = parser_handle.parse_args()

    alg_script = parser_handle.prog

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.alg_time_start:
        alg_time_start = parser_values.alg_time_start
    else:
        alg_time_start = None

    if parser_values.alg_time_end:
        alg_time_end = parser_values.alg_tim_end
    else:
        alg_time_end = None

    return alg_script, alg_settings

# -------------------------------------------------------------------------------------

# Call script from external library
if __name__ == "__main__":
    main()
    
# ----------------------------------------------------------------------------

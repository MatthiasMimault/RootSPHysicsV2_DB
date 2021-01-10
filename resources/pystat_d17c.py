# -*- coding: utf-8 -*-
"""
Release on 08/06/2020
0806 v10 - Return average along with the standard error (flawed with multip)
0608 v11 - Introduction of Beemster Data (flawed)
0109 v12 - Figure generation without display
0109 v13 - Beemster json data read, in-shell input
0110 v15 - Output Velocities
1410 v16 - Output Viscous forces (e Adimen)
2611 v17 - Beesmter Option, multicore version (loop independance)
@author: MM42910
"""

import numpy as np
import csv, os, time, sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### -1 Def functions
def jupdate(f, fd, ts):
    if not os.path.exists(f):
        with open(f, 'w') as jf:
            data = {fd:ts}
            json.dump(data,jf,indent=4)
            jf.close()
    else:
        with open(f, 'r+') as jf:
            data = json.load(jf)
            data[fd] = ts
            jf.seek(0)
            json.dump(data,jf,indent=4)
            jf.truncate()
            jf.close()

def stats(d):
    sig = dict.fromkeys(d.keys(), 0.0)    
    avg = dict.fromkeys(d.keys(), 0.0)  
    avg2 = dict.fromkeys(d.keys(), 0.0)
    for item in d.keys():
        loc_avg = 0;
        loc_avg2 = 0;
        n = 0;
        for val in d[item]:
            loc_avg += val;
            loc_avg2 += val**2;
            n += 1;
        if n != 0:
            avg[item] = loc_avg/n  
            avg2[item] = loc_avg2/n
    sig = [np.sqrt(abs(avg2[id]-avg[id]**2)/75.0) for id in d.keys()]
    return avg, sig


def procPos(l,b):
    maxPos = 0.0
    for item in l:
        # mm to um conversion
        item['Pos.x'] = float(item['Pos.x'])*1000
        maxPos = max(maxPos, item['Pos.x'])
    
    # Change to tip referential
    for item in l:
        item['Pos.x'] = abs(maxPos-item['Pos.x'])

    # Filter ptc out of bins
    l = [item for item in l if item['Pos.x']<b[-1]]
    
    #> Change to bin        
    for ptc in l:
        ptc['Pos.x'] = [pb for pb in b 
            if abs(pb-ptc['Pos.x'])<=binWidth/2][0]
    return l

def generPrevCount(name):
    d = {}
    with open(csvFolder+"/"+name+"_stats.csv") as csvfile:
        next(csvfile)
        next(csvfile)
        next(csvfile)
        dP = []
        rdP = csv.DictReader(csvfile, delimiter=";")
        for row in rdP:
            del row['']
            dP.append(row) 
    Dt = float(dP[1]['Time'])
    for save in dP[:-1]:
        d[csvFolder+"/"+name+"_"+str(dP.index(save)+1).zfill(4)+".csv"] = int(save['Count'])
    d[csvFolder+"/"+name+"_0000.csv"] = int(dP[0]['Count'])        
    return d, Dt

def generTimeChck(name, t0):
    with open(csvFolder+"/"+name+"_stats.csv") as csvfile:
        next(csvfile)
        next(csvfile)
        next(csvfile)
        dP = []
        rdP = csv.DictReader(csvfile, delimiter=";")
        for row in rdP:
            del row['']
            dP.append(row) 
    return [float(i['Time'])/t0/3600.0 for i in dP]

#######################################################################
### -1 Beemster Data

#######################################################################
### 0 Prelude
# Argument read - inline
arguments = sys.argv
caseShort = arguments[1]
csvFolder = arguments[2]
imgFolder = arguments[3]
options = arguments[4]
n_avg = int(arguments[5])

# Dimensions
x0 = 1000.0 # [m] = 10^3 [mm]
m0 = 1e3   # [kg] = 10^-3 [T]
t0 = 1e-3   # [s] = 10^-3 [1000 s]

rho0 = m0/x0**3.0
v0 = x0/t0
#P0 = 1e6*t0**2/(m0*x0) 
#L0 = rho0/t0

#Bin details
binWidth = 625;
binLength = 15000;

# Viscous coefficient
mu = 10

# Font size
SMALL_SIZE = 14
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


# File detection - Get Pure name
caseName = [item[:-10] for item in os.listdir(csvFolder)
    if caseShort in item and "stats" in item][0]
if not os.path.exists(imgFolder):
   os.mkdir(imgFolder)
if not os.path.exists(imgFolder+"/"+caseName+"-"+str(n_avg)):
    os.mkdir(imgFolder+"/"+caseName+"-"+str(n_avg))


opt_len = "l" in options
opt_div = "d" in options
opt_stn = "s" in options
opt_vel = "v" in options
opt_vsc = "f" in options
opt_bee = "b" in options

# Detection and read of Beemster 98 data    
if opt_bee:
    with open('Bee98Data.json') as json_file:
        B98 = json.load(json_file)
            
# Generate list of Csv names
CsvList = [name for name in os.listdir(csvFolder) 
    if caseName in name and 'stats' not in name and '.vtk' not in name]
CsvList.sort()
#CsvList = CsvList[:5]

# Generate initial Smoothing subset
bins = np.arange(binWidth/2, binLength+binWidth/2,binWidth)
smoSub = []

# Generate Time serie vector
TT = generTimeChck(caseName, t0)
#TT = TT[:5]
if opt_len == True:
    TLm = []
    TL1 = []
    TLM = []
if opt_stn == True:
    TSm = []
    TS1 = []
    TSM = []
if opt_div == True:
    TDm = []
    TD1 = []
    TDM = []
if opt_vel == True:
    TVm = []
    TV1 = []
    TVM = []
if opt_vsc == True:
    TFm = []
    TF1 = []
    TFst = []
    SV = []


tic = time.perf_counter()
tic_loop = time.perf_counter()
    
if opt_div == True:   
    prevCount, Dt = generPrevCount(caseName)


toc_loop = time.perf_counter()
#print(f"Generation smoothing subset in {toc_loop - tic_loop:0.4f} s")

### 1 loop
#> Dev - Loop independance for Multi-core
for i in np.arange(n_avg-1,len(CsvList)):
    ### 10 Initialise var
#    nb = [0] * len(bins)
    lenb = [[] for i in range(len(bins))]
    inb = [0 for i in range(len(bins))]
    divb = [0 for i in range(len(bins))]
    stnb = [[] for i in range(len(bins))]
    velb = [[] for i in range(len(bins))]
    vscb = [[] for i in range(len(bins))]
    tic_loop = time.perf_counter()
    
    ### 11 Load smoothing pack names
    for j in np.arange(i-n_avg+1,i+1):
        currentCaseName = csvFolder+"/"+CsvList[j]  
        smoSub.append(currentCaseName)
        
    # for loop data in dataSubset
    Fbar_temp = 0.0
    v_temp = 0.0
    mr_temp = 0.0
    sv_temp = 0.0
    
    for sub in smoSub:
        # read sub csv    	
        with open(sub) as csvfile:
            next(csvfile)
            next(csvfile)
            next(csvfile)
            dP = []
            rdP = csv.DictReader(csvfile, delimiter=";")
            for row in rdP:
                del row['']
                dP.append(row) 
                
        # process new csv          
        dP = procPos(dP, bins)
        
        for ptc in dP:
            if opt_len == True:
                lenb[np.where(bins==ptc['Pos.x'])[0][0]].insert(0,2.0*x0/np.sqrt(float(ptc['Qfxx'])))
            if opt_stn == True:
                stnb[np.where(bins==ptc['Pos.x'])[0][0]].insert(0,float(ptc['StrainDot.x'])*100.0*3.6)
            if opt_div == True:
                inb[np.where(bins==ptc['Pos.x'])[0][0]]+=1
                if int(ptc['Idp'])>=prevCount[sub]:
                    divb[np.where(bins==ptc['Pos.x'])[0][0]]+=1
            if opt_vel == True:
                # Bin collection of velocity value, scaled to chosen space and time scale (mm and hours)
                velb[np.where(bins==ptc['Pos.x'])[0][0]].insert(0,float(ptc['Vel.x'])*3.6)
            if opt_vsc == True:
                Fbar_temp += float(ptc['Mass'])*float(ptc['AceVisc.x'])/n_avg
                v_temp += float(ptc['Vel.x'])/n_avg/len(smoSub[sub])
                mr_temp += float(ptc['Mass'])/float(ptc['Rhop'])/n_avg
                sv_temp += float(ptc['Vel.x'])*2.0*x0/np.sqrt(float(ptc['Qfxx']))/n_avg
# 2 Process average
    if opt_len == True:
        len_avg, len_std = stats(dict(zip(bins,lenb)))
        TLm.append(min(len_avg.values()))
        TL1.append(sum(len_avg.values())/len(bins))
        TLM.append(max(len_avg.values()))
    if opt_stn == True:
        stn_avg, stn_std = stats(dict(zip(bins,stnb)))
        TSm.append(min(stn_avg.values()))
        TS1.append(sum(stn_avg.values())/len(bins))
        TSM.append(max(stn_avg.values()))
    if opt_div == True:
        inb = [1 if x==0 else x for x in inb]
        div_avg = [a/b/Dt for a,b in zip(divb,inb)]
        TDm.append(min(div_avg))
        TD1.append(sum(div_avg)/len(bins))
        TDM.append(max(div_avg))
    if opt_vel == True:
        vel_avg, vel_std = stats(dict(zip(bins,velb)))
        TVm.append(min(vel_avg.values()))
        TV1.append(sum(vel_avg.values())/len(bins))
        TVM.append(max(vel_avg.values()))
    if opt_vsc == True:
        TF1.append(Fbar_temp)
        TFst.append(mu*v_temp*mr_temp)
        SV.append(mu*sv_temp/(6.0*1e7))
    
    # 3 Plot stats
    # Cell length
    if opt_len == True:
        fig, ax = plt.subplots(1, figsize=(8, 6))
        fig.suptitle('Cell length')        
        ax.errorbar(bins,len_avg.values(), yerr= len_std, marker = 'o')
        if opt_bee:
            ax.errorbar(B98['xb_cl'], B98['yb_cl'], yerr = B98['eb_cl'], color = 'darkorange', marker = 'o')

        plt.xlim(0, binLength)    
        # plt.ylim(0, 85)     
        plt.xlabel(r'Distance from the tip ($\mathrm{\mu m}$)')
        plt.ylabel(r'Cell length ($\mathrm{\mu m}$)')  
        # Save data and figures
        plt.savefig(imgFolder+"/"+caseName+"-"+str(n_avg)+"/Len"+str(n_avg)+"-"+CsvList[i][:-4]+".png")
        # Clear figure
        plt.close()  
    
    # Strain rate
    if opt_stn == True:   
        fig, ax = plt.subplots(1, figsize=(8, 6))
        fig.suptitle('Strain rate')        
        ax.errorbar(bins, stn_avg.values(), yerr= stn_std, marker = 'o')
        if opt_bee:
            ax.errorbar(B98['xb_sr'], B98['yb_sr'], yerr= B98['eb_sr'], color = 'darkorange', marker = 'o')
        #plt.scatter(bins, stn_avg.values())
        #plt.errorbar(bins, stn_avg.values(), stn_std)
        plt.xlim(0, binLength)    
        plt.ylim(0, 60.0)   
        plt.xlabel(r'Distance from the tip ($\mathrm{\mu m}$)')
        plt.ylabel(r'Strain rate ($\%.\mathrm{h}^{-1}$)')  
        
        # Save data and figures
        plt.savefig(imgFolder+"/"+caseName+"-"+str(n_avg)+"/Stn"+str(n_avg)+"-"+CsvList[i][:-4]+".png")
        # Clear figure
        plt.close()  
    
    # Division rate
    if opt_div == True:
        fig, ax = plt.subplots(1, figsize=(8, 6))
        fig.suptitle('Cell division')        
        ax.plot(bins, div_avg, marker = 'o')
        if opt_bee:
            ax.errorbar(B98['xb_cd'], B98['yb_cd'], yerr = B98['eb_cd'], color = 'darkorange', marker = 'o')

        plt.xlim(0, binLength)    
        plt.ylim(0, 0.1)     
        plt.xlabel(r'Distance from the tip ($\mathrm{\mu m}$)')
        plt.ylabel(r'Division rate ($\mathrm{cell.cell}^{-1}.\mathrm{h}^{-1}$)')  

        # Save data and figures
        plt.savefig(imgFolder+"/"+caseName+"-"+str(n_avg)+"/Div"+str(n_avg)+"-"+CsvList[i][:-4]+".png")
        # Clear figure
        plt.close()  
    
    # Velocity
    if opt_vel == True:   
        fig, ax = plt.subplots(1, figsize=(8, 6))
        fig.suptitle('Velocity')        
        ax.errorbar(bins, vel_avg.values(), yerr= vel_std, marker = 'o')
#        ax.errorbar(B98['xb_sr'], B98['yb_sr'], yerr= B98['eb_sr'], color = 'darkorange', marker = 'o')
        plt.xlim(0, binLength)    
        #plt.ylim(0, 1)   
        plt.xlabel(r'Distance from the tip ($\mathrm{\mu m}$)')
        plt.ylabel(r'Velocity ($\mathrm{mm.h}^{-1}$)')  
        
        # Save data and figures
        plt.savefig(imgFolder+"/"+caseName+"-"+str(n_avg)+"/Vel"+str(n_avg)+"-"+CsvList[i][:-4]+".png")
        # Clear figure
        plt.close()  
    
    # Viscous force/acceleration
#    if opt_vsc == True:   
#        fig, ax = plt.subplots(1, figsize=(8, 6))
#        fig.suptitle('Viscous pressure')    
#        ax.plot(bins, vsc_avg, marker = 'o')    
##        ax.errorbar(B98['xb_sr'], B98['yb_sr'], yerr= B98['eb_sr'], color = 'darkorange', marker = 'o')
#        plt.xlim(0, binLength)    
#        plt.ylim(0, 20)   
#        plt.xlabel(r'Distance from the tip ($\mathrm{\mu m}$)')
#        plt.ylabel(r'Viscous pressure ($\mathrm{\mu g.mm.H^{-2}}$)')  
#        
#        # Save data and figures
#        plt.savefig(imgFolder+"/"+caseName+"-"+str(n_avg)+"/Vsc"+str(n_avg)+"-"+currentCaseName[4:-4]+".png")
#        # Clear figure
#        plt.close()  
    
    
    # 4 End of loop
    smoSub = []
    toc_loop = time.perf_counter()
    print(CsvList[i][:-4]+f" processed in  {toc_loop - tic_loop:0.4f} s")

# Time series plot
if opt_len == True:
    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.suptitle(caseName+' time series of Cell length', fontsize=15) 
    print('TT ', TT)
    print('TLm ', TLm)
    ax.plot(TT[n_avg-1:], TLm, label='Min') 
    ax.plot(TT[n_avg-1:], TL1, label='Mean')      
    ax.plot(TT[n_avg-1:], TLM, label='Max')       
    ax.legend()
    plt.xlabel('Time (H)')
    plt.ylabel('Cell length (μm)')  
    # Save data and figures
    plt.savefig(imgFolder+"/"+caseName+"-"+str(n_avg)+"/0TS-Len"+str(n_avg)+"-"+caseName+".png")
    # Clear figure
    plt.clf()  
    plt.close()  
if opt_stn == True:
    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.suptitle(caseName+' time series of Strain rate', fontsize=15)           
    ax.plot(TT[n_avg-1:], TSm, label='Min') 
    ax.plot(TT[n_avg-1:], TS1, label='Mean')      
    ax.plot(TT[n_avg-1:], TSM, label='Max')   
    ax.legend()
    plt.xlabel('Time (H)')
    plt.ylabel('Strain Rate (%.h^-1)')  
    # Save data and figures
    plt.ylim(max(0,min(TSm)), min(max(TSM),30))     
    plt.savefig(imgFolder+"/"+caseName+"-"+str(n_avg)+"/0TS-Stn"+str(n_avg)+"-"+caseName+".png")
    # Clear figure
    plt.clf() 
    plt.close()   
if opt_div == True:
    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.suptitle(caseName+' time series of Division rate', fontsize=15)           
    ax.plot(TT[n_avg-1:], TDm, label='Min') 
    ax.plot(TT[n_avg-1:], TD1, label='Mean')      
    ax.plot(TT[n_avg-1:], TDM, label='Max')   
    ax.legend()
    plt.xlabel('Time (H)')
    plt.ylabel('Division rate (cell.cell^-1.h^-1)')   
    # Save data and figures
    plt.savefig(imgFolder+"/"+caseName+"-"+str(n_avg)+"/0TS-Div"+str(n_avg)+"-"+caseName+".png")
    # Clear figure
    plt.clf()  
    plt.close()  
if opt_vel == True:
    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.suptitle(caseName+' time series of Velocity', fontsize=15)           
    ax.plot(TT[n_avg-1:], TVm, label='Min') 
    ax.plot(TT[n_avg-1:], TV1, label='Mean')      
    ax.plot(TT[n_avg-1:], TVM, label='Max')   
    ax.legend()
    plt.xlabel(r'Time (h)')  
    plt.ylabel(r'Velocity ($\mathrm{mm.h}^{-1}$)')    
    # Save data and figures
    plt.savefig(imgFolder+"/"+caseName+"-"+str(n_avg)+"/0TS-Vel"+str(n_avg)+"-"+caseName+".png")
    # Clear figure
    plt.clf() 
    plt.close()   
    
    #Write time series
    jupdate('TS-VEL.json', caseName, TV1)
    
if opt_vsc == True:
    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.suptitle(caseName+' time series of total viscous forces', fontsize=15)   
    ax.plot(TT[n_avg-1:], SV, label='Total pressure')  
    plt.xlabel(r'Time ($\mathrm{h}$)')
    plt.ylabel(r'Resisting pressure $\mathrm{Pa}$')   
    ax.legend() 
    plt.ylim(0, 1e-4)      
    # Save data and figures
    plt.savefig(imgFolder+"/"+caseName+"-"+str(n_avg)+"/0TS-Vsc"+str(n_avg)+"-"+caseName+".png")
    # Clear figure
    plt.clf() 
    plt.close()   
    
    #Write time series
    jupdate('TS-VSC.json', caseName, TF1)
    

### End of script
toc = time.perf_counter()
print(f"Processed achieved in  {toc - tic:0.4f} s")  

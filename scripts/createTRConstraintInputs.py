import pandas
import sys, getopt
import numpy as np
import Levenshtein

def getLPSMotifVariability(row):
    mcMotif=row['mostCommonLPSMotif']
    thisMotif=row['LPS_Motif']
    weight=row['N_motif']
    lpsMotifVariability=0
    if len(thisMotif)==len(mcMotif):
        dist=Levenshtein.distance(thisMotif,mcMotif)
        normDist=dist/len(thisMotif)
        lpsMotifVariability=lpsMotifVariability+(normDist*weight)
    elif len(thisMotif)<len(mcMotif):
        # calculate multiple
        mult=np.round(len(mcMotif)/len(thisMotif))
        extendedMotif=thisMotif*int(mult)
        dist=Levenshtein.distance(extendedMotif,mcMotif)
        normDist=dist/len(extendedMotif)
        lpsMotifVariability=lpsMotifVariability+(normDist*weight)
    elif len(thisMotif)>len(mcMotif):
        mult=np.round(len(thisMotif)/len(mcMotif))
        extendedMCMotif=mcMotif*int(mult)
        dist=Levenshtein.distance(thisMotif,extendedMCMotif)
        normDist=dist/len(thisMotif)
        lpsMotifVariability=lpsMotifVariability+(normDist*weight)
    else:
        print('Something went wrong')
    return lpsMotifVariability

def main ( argv ):
    lpsPerMotifFileName=""
    lpsMAFileName=""
    cpsFileName=""
    alleleLenFileName=""
    genicBedFileName=""
    intergenicBedFileName=""
    try:
        opts, args = getopt.getopt(argv,"h",["sample=","lpsPerMotif=","lpsMA=","cps=","alleleLen=","genicBed=","intergenicBed=","help"])
    except getopt.GetoptError:
        print('computeTRStats.py --lpsPerMotif=<InputFile1> --lpsMA=<InputFile2> --cps=<InputFile3> --alleleLen=<InputFile4> --genicBed=<BedFile> --intergenicBed=<BedFile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('--lpsPerMotif'):
            lpsPerMotifFileName=arg
        elif opt in ('--lpsMA'):
            lpsMAFileName=arg
        elif opt in ('--cps'):
            cpsFileName=arg
        elif opt in ('--alleleLen'):
            alleleLenFileName=arg
        elif opt in ('--genicBed'):
            genicBedFileName=arg
        elif opt in ('--intergenicBed'):
            intergenicBedFileName=arg
        elif opt in ('h','--help'):
            print('computeTRStats.py --lpsPerMotif=<InputFile1> --lpsMA=<InputFile2> --cps=<InputFile3> --alleleLen=<InputFile4> --genicBed=<BedFile> --intergenicBed=<BedFile>')
            sys.exit()
        else:
            print('computeTRStats.py --lpsPerMotif=<InputFile1> --lpsMA=<InputFile2> --cps=<InputFile3> --alleleLen=<InputFile4> --genicBed=<BedFile> --intergenicBed=<BedFile>')
            sys.exit()

    # read input data
    lpsPerMotif=pandas.read_csv(lpsPerMotifFileName,sep='\t')
    lpsMotifAgnostic=pandas.read_csv(lpsMAFileName,sep='\t')
    cps=pandas.read_csv(cpsFileName,sep='\t')
    alleleLen=pandas.read_csv(alleleLenFileName,sep='\t')

    # compute LPS Motif Variability
    tmp=lpsPerMotif.loc[:,['TRID','LPS_Motif']].groupby('TRID').count()
    tmp=tmp.rename(columns={'LPS_Motif':'numUniqueMotifs'})
    lpsVariability=lpsPerMotif.merge(tmp,how='inner',left_on='TRID',right_index=True)
    mostCommonMotif=lpsVariability.sort_values(by=['TRID','N_motif'],ascending=False).drop_duplicates(subset='TRID',keep='first').reset_index(drop=True)
    mostCommonMotif=mostCommonMotif.rename(columns={'LPS_Motif':'mostCommonLPSMotif'})
    lpsVariability=lpsVariability.merge(mostCommonMotif.loc[:,['TRID','mostCommonLPSMotif']],how='inner',on='TRID')
    lpsVariability['lpsMotifVariability']=lpsVariability.apply(getLPSMotifVariability,axis=1)
    tmp=lpsVariability.loc[:,['TRID','lpsMotifVariability','N_motif']].groupby('TRID').sum()
    tmp=tmp.rename(columns={'N_motif':'N'})
    tmp['lpsMotifVariability']=tmp.loc[:,'lpsMotifVariability']/tmp.loc[:,'N']
    tmp=tmp.drop(columns=['N'])
    lpsVariability=lpsVariability.drop(columns=['lpsMotifVariability'])
    lpsVariability=lpsVariability.merge(tmp,how='inner',on='TRID')
    lpsVariability=lpsVariability.loc[lpsVariability['50thPercentile']>0,:]
    lpsVariability=lpsVariability.sort_values(by=['TRID','N_motif'],ascending=False).drop_duplicates(subset='TRID',keep='first')
    lpsVariability=lpsVariability.loc[:,['TRID','numUniqueMotifs','lpsMotifVariability']]
    lpsVariability=lpsVariability.fillna(0)


    # select the appropriate columns and rename them
    mcLPSStats=lpsPerMotif.sort_values(by=['TRID','N_motif'],ascending=False).drop_duplicates(subset='TRID',keep='first').reset_index(drop=True)
    mcLPSStats=mcLPSStats.rename(columns={'50thPercentile':'mclpsmMedian','mean':'mclpsmMean','std':'mclpsmStdev','mad':'mclpsmMAD'}).loc[:,['TRID','mclpsmMedian','mclpsmMean','mclpsmStdev','mclpsmMAD']]
    lpsMotifAgnostic2=lpsMotifAgnostic.rename(columns={'50thPercentile':'combinedLPSMedian','mean':'combinedLPSMean','std':'combinedLPSStdev','mad':'combinedLPSMAD'}).loc[:,['TRID','combinedLPSMedian','combinedLPSMean','combinedLPSStdev','combinedLPSMAD']]
    alleleLen2=alleleLen.rename(columns={'50thPercentile':'totalLengthMedian','mean':'totalLengthMean','std':'totalLengthStdev','mad':'totalLengthMAD'}).loc[:,['TRID','totalLengthMedian','totalLengthMean','totalLengthStdev','totalLengthMAD']]   

    # merge the desired annotations together
    combined=mcLPSStats.merge(lpsMotifAgnostic2,how='inner',on='TRID')
    combined=combined.merge(alleleLen2,how='inner',on='TRID')
    combined=combined.merge(cps.loc[:,['TRID','CPS']],how='inner',on='TRID')
    combined=combined.merge(lpsVariability,how='inner',on='TRID')
    combined=combined.fillna(0)
    combined=combined.loc[:,['TRID','mclpsmMedian','mclpsmMAD','mclpsmMean','mclpsmStdev','numUniqueMotifs','lpsMotifVariability',
                             'totalLengthMedian','totalLengthMAD','totalLengthMean','totalLengthStdev',
                             'combinedLPSMedian','combinedLPSMAD','combinedLPSMean','combinedLPSStdev','CPS']]
    
    #### merge onto genic and intergenic bed files
    # load genic and intergenic bed files
    genic10kb=pandas.read_csv(genicBedFileName,sep='\t',header=None,names=['chr','start','end','INFO'])
    intergenic10kb=pandas.read_csv(intergenicBedFileName,sep='\t',header=None,names=['chr','start','end','INFO'])

    # remove sites where total span is under 10bp of repeat
    intergenic10kb=intergenic10kb.loc[(intergenic10kb['end']-intergenic10kb['start'])>=9,:].reset_index(drop=True)
    genic10kb=genic10kb.loc[(genic10kb['end']-genic10kb['start'])>=9,:].reset_index(drop=True)
    # remove sites on chrX or chrY
    intergenic10kb=intergenic10kb.loc[~(intergenic10kb['chr'].isin(['chrX','chrY'])),:].reset_index(drop=True)
    genic10kb=genic10kb.loc[~(genic10kb['chr'].isin(['chrX','chrY'])),:].reset_index(drop=True)
    # add TRID column
    intergenic10kb['TRID']=intergenic10kb.loc[:,'INFO'].str.split(';',expand=True).iloc[:,0].str[3:]
    intergenic10kb=intergenic10kb.loc[:,['chr','start','end','TRID']]
    genic10kb['TRID']=genic10kb.loc[:,'INFO'].str.split(';',expand=True).iloc[:,0].str[3:]
    genic10kb=genic10kb.loc[:,['chr','start','end','TRID']]
    # add on variation information
    intergenic10kb=intergenic10kb.merge(combined,how='inner',on='TRID')
    genic10kb=genic10kb.merge(combined,how='inner',on='TRID')

    # split intergenic bed file into training and validation sets
    indices=np.random.choice(np.arange(len(intergenic10kb)),size=np.floor(0.9*len(intergenic10kb)).astype(int),replace=False)
    intergenic10kbTrain = intergenic10kb.loc[intergenic10kb.index.isin(list(indices)),:]
    intergenic10kbVal = intergenic10kb.loc[~(intergenic10kb.index.isin(list(indices))),:]

    # write output
    intergenic10kbTrain.to_csv('training_10kbFromGenes.bed',sep='\t',index=False)
    intergenic10kbVal.to_csv('validation_10kbFromGenes.bed',sep='\t',index=False)
    genic10kb.to_csv('testing_within10kbOfGenes.bed',sep='\t',index=False)
    return


if __name__ == "__main__":
    main(sys.argv[1:])

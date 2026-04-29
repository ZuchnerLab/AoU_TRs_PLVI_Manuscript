import pandas
import sys
import json

def main ( argv ):
    samples=pandas.read_csv('srWGS_bam_list.txt',sep='\t',header=None,names=['person_id', 'bam_path'])
    fEP=open('EP400_EH_AlleleLengths_245381_samples.txt','w')
    fFAM=open('FAM193B_EH_AlleleLengths_245381_samples.txt','w')
    count=0
    for sampleID in samples.loc[:,'person_id'].values:
        if ((count % 1000) == 0):
            print('Working on sample ' + str(count))
        count=count+1
        thisSample=json.load(open('EH_results/wgs_' + str(sampleID)[6:13] + '.json','r'))
        if ('Genotype' in thisSample['LocusResults']['EP400']['Variants']['EP400'].keys()):
            genotypes=thisSample['LocusResults']['EP400']['Variants']['EP400']['Genotype'].split('/')
            for geno in genotypes:
                fEP.write(geno + '\n')
        if ('Genotype' in thisSample['LocusResults']['FAM193B']['Variants']['FAM193B'].keys()):
            genotypes=thisSample['LocusResults']['FAM193B']['Variants']['FAM193B']['Genotype'].split('/')
            for geno in genotypes:
                fFAM.write(geno + '\n')
    fEP.close()
    fFAM.close()
    return

if __name__ == "__main__":
    main(sys.argv[1:])

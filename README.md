# HSFSurv

## Usage
### Data Acquisition and Preprocessing Procedures

1) Diagnostic whole-slide images (WSIs) were obtained from five [TCGA](https://portal.gdc.cancer.gov) cohorts:
 TCGA-LUAD, TCGA-KIRC, TCGA-BLCA, TCGA-STAD, and TCGA-COAD. 
2) Foreground tissue regions were extracted at 40× magnification using the WSI processing utilities provided in [CLAM](https://github.com/mahmoodlab/CLAM),
 followed by segmentation into non-overlapping 512×512 pixel patches.
3) Patch-level feature representations were computed using a ResNet-50 model pretrained on ImageNet.
   The extracted features for each WSI were saved individually as .pt files, which were then organized into directories to serve as the input features for the pathology modality.
    Pathological image preprocessing was conducted following the protocol established in the CLAM framework.
4) Genomic and clinical survival data were retrieved from the [cBioPortal](https://www.cbioportal.org) database.
5) The downloaded file data clinical patient.txt contains patient clinical information.
Prior to executing the code, it is necessary to convert this file into CSV format to ensure compatibility with
the analysis pipeline. Samples with missing mRNA expression profiles or overall
survival records were excluded from the analysis.

### Run code
```
python main.py --omic_modal mRNA --kfold_split_seed 42 --pretrain_epochs 20 --finetune_epochs 40 --model_fusion_type concat --model_pretrain_fusion_type concat --cuda_device 0 --experiment_id 0 --seperate_test
```

## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [CLAM](https://github.com/mahmoodlab/CLAM)
- [MCAT](https://github.com/mahmoodlab/MCAT)
- [PathOmics](https://github.com/Cassie07/PathOmics)
- [CMTA](https://github.com/FT-ZHOU-ZZZ/CMTA)
- [CCL](https://github.com/moothes/CCL-survival)
- [SurvPath](https://github.com/mahmoodlab/SurvPath)
## License
This code is available for non-commercial academic purposes.
## HRF Segmentation on OCTs

2D segmentation (per B-scan) of Hyperreflective Foci (HRF) on OCT images using the nnSAM model.

*A-EYE Unit, Wisconsin Reading Center, Department of Ophthalmology and Visual Sciences, UW–Madison*

### Pipeline

1. **Preprocessing** — OCT volumes separated per subject into individual B-scan slices, each paired with its corresponding mask.
2. **Data organization** — files structured into the format required by nnSAM.
3. **B-scan selection** — for every positive B-scan, an equal number of neighboring B-scans (before/after in the same subject) were sampled as negatives, ensuring adequate negative representation without letting negatives dominate the dataset.
4. **Model run** — nnSAM run on the selected B-scans, with results evaluated on both internal and external test sets

### Status

- ✅ OCT preprocessing — per-subject, per-slice organization with paired masks
- ✅ Data reformatted into nnSAM-compatible structure
- ✅ B-scan sampling strategy implemented (balanced positive/negative selection)
- ✅ nnSAM run on internal and external test sets
- 🔄 **In progress:** results are being reviewed by graders, and the model may be refined further as a result — performance metrics are not being shared publicly at this time.

<img width="1547" height="332" alt="image" src="https://github.com/user-attachments/assets/0c6ced02-3c7b-4d9c-a844-42d5bda9f579" />



    
*Note: The image above is NOT from our private dataset and is used here for demonstration purposes ONLY.*

     

### Contact

Reeva Faisal — [rfaisal@wisc.edu](mailto:rfaisal@wisc.edu)

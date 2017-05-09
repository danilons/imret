# Image Retrieval Region Connection Calculus

Steps necessary to run:

1. Create segmentation labels
1. Train segmentation model
1. Segment images
1. Create topology database (optional)
1. Create preposition base
1. Train preposition model
1. Create index


## Evaluation

In parallel, there are some scripts to evaluate the process:

1. eval_iou
    Evaluate segmentation

1. eval_preposition_model
    Evaluate preposition estimation

1. eval_queries
    Evaluate retrieval

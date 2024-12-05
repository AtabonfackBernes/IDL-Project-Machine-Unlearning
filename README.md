# IDL-Project: Saliency-Unlearning With Integrated Gradient

**Contact**: Saliency-Unlearning With Integrated Gradient

**Contributors**:
* Atabonfack Bernes {batabonf@andrew.cmu.edu }
* Olatunji Damilare E. {dolatunj@andrew.cmu.edu}  
* Pierre Ntakirutimana  {pntakiru@andrew.cmu.edu}   
* Tanghang Elvis Tata  {etanghan@andrew.cmu.edu}


## General concept: 






## Abstract: 

Abstract
The field of machine learning is rapidly growing. It has been used to solve several social and
engineering problems, image recognition, text and image generations,... The data governance issues
and data protection policies were built to make the field fair. Subjects deserve the right to withdraw
their consent and ask their data to be removed from models. The unlearning techniques help in
removing the dataset with minimum changes to the model. We presented an integrated gradient
saliency unlearning architecture that integrates the gradient of the model over a whole path and
thresholds it to deal with the contribution of the forget dataset on the entire model path weights

### Our Idea: 
We propose an adaptive gradient saliency framework that enhances machine unlearning for multi-class image classification
models to address this gap. Our research aims to identify and modulate the most salient gradients
without incurring the computational burden of full retraining or sacrificing the model’s classification
performance on non-targeted classes. Through rigorous evaluation, we aim to justify that adaptive
gradient saliency offers a scalable and robust solution for machine unlearning. We also aim to advance
the standards of data-driven model management in an era that increasingly values privacy, agility, and
ethical AI practices.

<table align="center">
  <tr>
    <td align="center"> 
      <img src="images/image.png" alt="Teaser" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 2:</strong>  Overview of our machine unlearning technique (SalUn).</em>
    </td>
  </tr>
</table>

    
The code structure of this project is adapted from the https://github.com/OPTML-Group/Unlearn-Saliency codebase.
For this project we Added a new script **new_generate_mask.py** that applies integrated gradient to obtain the salient weigts and biases used during unlearning 

## Requirements
```bash
pip install -r requirements.txt
```

## Scripts
1. Get the origin model.
    ```bash
    python main_train.py --arch {model name} --dataset {dataset name} --epochs {epochs for training} --lr {learning rate for training} --save_dir {file to save the orgin model}
    ```

    A simple example for ResNet-18 on CIFAR-10.
    ```bash
    python main_train.py --arch resnet18 --dataset cifar10 --lr 0.013 --epochs 182
    ```

2. Generate Saliency Map
    ```bash
    python new_generate_mask.py --save_dir ${saliency_map_path} --model_path ${origin_model_path} --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs 1
    ```

3. Unlearn
    *  InSalUn
    ```bash
    python main_random.py --unlearn RL --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --num_indexes_to_replace ${forgetting data amount} --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path ${saliency_map_path}
    ```

    A simple example for ResNet-18 on CIFAR-10 to unlearn 10% data.
    ```bash
    python main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.013 --num_indexes_to_replace 4500 --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path mask/with_0.5.pt
    ```

    To compute UA, we need to subtract the forget accuracy from 100 in the evaluation results. As for MIA, it corresponds to multiplying SVC_MIA_forget_efficacy['confidence'] by 100 in the evaluation results. For a detailed clarification on MIA, please refer to Appendix C.3 at the following link: https://arxiv.org/abs/2304.04934.


    * Retrain
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn retrain --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

    * FT
    ```bash
    python main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn FT --num_indexes_to_replace ${forgetting data amount} --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning}
    ```

   * FT with Integrated Saliency
    ```bash
   python main_random.py --unlearn FT --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --num_indexes_to_replace ${forgetting data amount} --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path ${saliency_map_path}
    ```

    * IU
    ```bash
    python -u main_forget.py --save_dir ${save_dir} --model_path ${origin_model_path} --unlearn wfisher --num_indexes_to_replace ${forgetting data amount} --alpha ${alpha}
    ```

   * IU with Integrated Saliency
    ```bash
     python main_random.py --unlearn wfisher --unlearn_epochs ${epochs for unlearning} --unlearn_lr ${learning rate for unlearning} --num_indexes_to_replace ${forgetting data amount} --model_path ${origin_model_path} --save_dir ${save_dir} --mask_path ${saliency_map_path} --alpha ${alpha}



# Extra material on how to use Integrated Gradients (IG)
Integrated Gradients is a systematic technique that attributes a deep model's prediction to its base features. For instance, an object recognition network's prediction to its pixels or a sentiment model's prediction to individual words in the sentence.The technique is based on the [paper](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) at ICML'17, a top tier machine learning conference.

[Variants ](https://arxiv.org/abs/1805.12233)of IG can be used to apply the notion of attribution to neurons.

That said, IG does not uncover the logic used by the network to combine features, though there are variants of IG that can do this in a limited sense.

*  Implementation of integration via summing the gradients is well explained in the paper [paper](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf).)


## Cite This Work
```
@article{fan2023salun,
  title={Saliency-Unlearning With Integrated Gradient},
  author={Atabonfack Bernes and Tanghang Elvis Tata and Olatunji Damilare E and Pierre Ntakirutimana},
  journal={},
  year={}
}
```

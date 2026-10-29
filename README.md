# Image-Based Eye-Drop Robot

[日本語](README.ja.md) | **English**

This project explores using camera images to control a robot arm that handles a conventional eye-drop bottle. It is intended as a study of how a robot might assist people who have difficulty applying eye drops themselves. The person, robot, bottle, and drop are all represented **in simulation**.

We developed a fixed-procedure demonstration and used it to fine-tune an existing robot AI model. No experiments were conducted on a physical robot or a human subject.

![Initial scene of a fixed-procedure demonstration](docs/teacher-start.png)

## How it works

The robot receives images from two cameras, the state of its arm and fingers, and a shared task instruction. It outputs arm movements and the finger width used to press the bottle. Ground-truth eye coordinates, the person's identity, and the bottle's internal drop state are not provided as AI inputs.

- **Simulation**: PyBullet and Three.js
- **Robot learning**: [LeRobot](https://github.com/huggingface/lerobot) and the pretrained [SmolVLA](https://huggingface.co/lerobot/smolvla_base) model
- **Training data**: 12 successful fixed-procedure demonstrations, split into 6 for training and 6 for testing
- **Task**: approach the eye, wait until it is open, press the bottle to release one drop, open the fingers, and retreat

The bottle is fixed to the arm; pressing and drop release are represented by simplified models.

## Results

We compared the fixed procedure with the AI model before and after the latest fine-tuning, using two character models that had previously been used during development. Each method was tested once per character.

| Method | Character model A | Character model B |
| --- | --- | --- |
| Fixed-procedure demonstrator | Released one drop, released pressure, and retreated | Released one drop, released pressure, and retreated |
| AI: before latest fine-tuning | Did not release a drop | Did not release a drop |
| AI: after latest fine-tuning | Stopped at 0.25 seconds after an out-of-range finger-width command | Pressing was not sustained during the 60-second simulation; no drop was released |

The pre-fine-tuning comparison used weights that had already been updated 600 times from the public pretrained model. The post-fine-tuning weights received another 3,000 updates using the demonstrations in this experiment. These are single trials for each condition; they do not establish a general success rate or performance on previously unseen characters.

## Repository contents

| Path | Contents |
| --- | --- |
| `scripts/` | Code used to collect demonstrations, prepare data, train, and evaluate |
| `outputs/runtime/review-sklearn-v1/src/` | Code for face recognition, physics simulation, and control |
| `preview/` | Code and assets for rendering characters, the arm, and the bottle |
| `source-deltas/` | Changes to image recording and aggregation made after the experiment |
| [CREDITS.md](CREDITS.md) | Asset sources and usage terms, in Japanese |
| [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) | Provenance and verification status of the archived and supplemented code, in Japanese |

Character model A is pixiv's VRM1_Constraint_Twist_Sample, model B is Seed-san, and the third model used for training is Godette. The identifier `original` in the source and records refers to model A.

## Environment and checks

The experiments used an Apple Silicon Mac, Python 3.12, PyTorch 2.11, LeRobot 0.6.1, PyBullet 3.2.7, scikit-learn 1.8.0, and other dependencies.

**The demonstration data, position estimator, weights before and after fine-tuning, and random states are not included in this repository.** Reproduction of the reported results from this repository alone has therefore not been verified. The conditions and method for distributing the data and weights have not been established. The public `smolvla_base` model alone does not provide the previously updated weights used for the comparison.

In the existing Mac environment, we verified that the main Python modules for collection, training, and evaluation could be imported. Some unit tests for the rendering-side code can be run as follows:

```sh
cd preview
npm ci --ignore-scripts
node --test anatomy-contact.test.mjs anatomy-score.test.mjs learning-conditions.test.mjs
```

These checks cover 13 unit tests; they do not establish that the complete eye-drop task can be reproduced or run in another environment.

## Future work

The AI must learn to maintain a stable finger width and complete the sequence from waiting for an open eye through pressing, releasing, and retreating. Performance with other characters, actual bottles and liquids, and real-time operation has not been evaluated.

For asset sources and usage terms, see [CREDITS.md](CREDITS.md) (in Japanese).

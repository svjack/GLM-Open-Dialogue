<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">GLM-Open-Dialogue</h3>

  <p align="center">
   		A enhanced Open Dialogue Context Generator supported by General Language Model Pretraining with Autoregressive Blank Infilling
    <br />
  </p>
</p>

## Brief introduction
Given the definition of Open-Domain Dialogue Generation [Open-domain Dialogue Generation:
What We Can Do, Cannot Do, And Should Do Next](https://aclanthology.org/2022.nlp4convai-1.13.pdf)
```
Given zero or more previous dialogue turns between
itself and one or more other participants,
a system  output a fluent, engaging,
and meaningful natural language response.
```
A well known model deal with this task is [DialoGPT](https://github.com/microsoft/DialoGPT),
You can try [DialoGPT](https://github.com/microsoft/DialoGPT) on [Roxza DialoGPT space](https://huggingface.co/spaces/Roxza/DialoGPT)

Following is a English example compare the output of DialoGPT and GLM(this project do)

### English Examples

|Model |First Question | Response chain |
|---------|--------|-------|
|DialoGPT |Does money buy happiness? | Depends how much money you spend on it.|
|||  What is the best way to buy happiness ?|
|||  You just have to be a millionaire by your early 20s, then you can be happy .|
||| This is so difficult !|
|||You have no idea how hard it is to be a millionaire and happy . There is a reason the rich have a lot of money|
|GLM |Does money buy happiness? | Can money buy happiness?|
|||Yes is.How about you?|
|||It is happiness, ha.|
|||If you fight for yourself, you will be happy.|
|||Oh, I got it.How about you?|
|||and happiness.|
|||Everyone has a different definition of happiness.|
|||Money cannot buy happiness, money can buy happiness.|
|DialoGPT |what is the meaning of life ? | The  meaning  is  to  be  with  your  family  and friends .|
|||I’m going to guess :  It means that your parents and friends have loved you . That’s the meaning , I believe .|
|||I think  the  meaning  is  to  make  sure  you’re alive when you die . So you’ll be with your family , friends , and yourself .|
|||It’s not , but thanks for trying to help .|
GLM |what is the meaning of life ? | You just like life.|
|||The answer, of course, is life itself.|
|||Has anyone said that life has no meaning?|
|||Yes you are right.What do I like the most?|
|||Do something meaningful.|
|||Enjoy every moment of your life and feel meaningful.|

## HuggingFace Space demonstration
* 1 Multilanguage demonstration: https://huggingface.co/spaces/svjack/GLM-Open-Dialogue
* 2 Chinese demonstration: https://huggingface.co/spaces/svjack/GLM-Open-Dialogue-Chinese

## Installation and Instructions
```bash
pip install -r req0.txt
pip install -r req1.txt
```
### Instructions
```python
from main import *

#### English Usage
demo_func(*["Which keyboard do you prefer?", "en", 2])
demo_func(*["Which keyboard do you prefer?", "en", 10])
#### Japanese Usage
demo_func(*["東京と京都のどちらが住みやすいですか。", "ja", 2])
demo_func(*["東京と京都のどちらが住みやすいですか。", "ja", 10])

#### Chinese Usage
demo_func(*["程序员要掌握哪些技能", "zh", 5])
### OR
generate_func(*["程序员要掌握哪些技能", 5])

generate_func(*["你对《三国演义》感兴趣吗?", 10])
generate_func(*["谁能让美国再一次伟大?", 10])
```

## Construction
### Migrate GLM to Open-Domain Dialogue Generation domain
Many this kinds models focus train a generate model on some specific datasets, This may constraints from datasets and pretrained models, When faced with a open domain with a broad breadth of knowledge, the generator may has more risk of cause Blandness Problem by bland answers or give some response lack of vividness. </br>
This project deal with this kinds of problems by a unsupervised combination method. And decompose this problem to three steps:
* 1 Self-train a Open-Domain Dialogue generator on a small dataset from scratch and use it as a <b>"Construct Generator"</b>
* 2 Use [GLM](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/glm_blank_filling/glm_generate_samples.py) as <b>"Knowledge Generator"</b>
* 3 Use a self-trained <b>"Context Discriminer"</b> to combine above two generators. <br/>

There is no high requirement for the accuracy of the "Construct Generator".<br/>

The above three models are released in [HuggingFace](https://huggingface.co), list in the below sheet.

|Role in Project |HuggingFace link| Contributor |
|---------|--------|-------|
|Construct Generator| https://huggingface.co/svjack/T5-daliy-dialogue | https://huggingface.co/svjack |
|Knowledge Generator| https://huggingface.co/BAAI/glm-large-chinese | https://huggingface.co/BAAI |
|Context Discriminer| https://huggingface.co/svjack/T5-dialogue-choose | https://huggingface.co/svjack |

They all build above Chinese, and use [facebook m2m_100_1.2B](https://huggingface.co/facebook/m2m100_1.2B) translator make them support Multilanguage domain.

### Prompt --- basic concept throughout the project
This are three entry points <b>Prompt</b> make for this project.

|Prompt category|Effect parts|Idea source|Prompt format in English|Prompt format in Chinese|
|---------|--------|-------|--------|-------|
|Task Oriented Prompt Format for QA Blank Infilling|[GLM](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/glm_blank_filling/glm_generate_samples.py)|build in GLM|Question: Is reading good for your grow up? Answer: [MASK]|问题: 阅读对于成长有好处吗？ 回答: [MASK]|
|Task Oriented Prompt Format for choose best|[Context Discriminer](https://huggingface.co/svjack/T5-dialogue-choose)|[Bigscience-mt0](https://huggingface.co/bigscience/mt0-large)|According to the following context, choose the optimal follow-up sentence Context: {}\n{}Answer:|根据如下上下文，选择最优的后续句子 上下文：{}\n{}答案：|
|Quality Oriented Negative Prompt|negative_prompts in [predict_t5_choose_simple.py](predict_t5_choose_simple.py)|[Stable Diffusion](https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/)|[Yes, I don't know, Who knows]|[是的,怎么知道]|



## Examples

|Language |First Question | Response chain |
|---------|--------|-------|
|English |Which keyboard do you like? | My true love!|
|||My favorite mechanical keyboard!|
|||The mechanics.keyboard|
|||HP Mechanical Keyboard|
|||Of course a mechanical keyboard.|
|||I like the mechanical keyboard, though not the hippie mouse.|
|||I like it, but listen.Do you want?|
|||HP's mechanical keyboard is really useful, and the cost is very high, but|
|Japanese |東京と京都はどちらがいいですか?|はい。,..わからないけど、あなたはどうですか?|
|||京都!|
|||うん、確信がない。,..あなたはどうですか?|
|||京都に行くと東京は観光都市です。|
|||誰もが京都よりはマシだと思っている。また、日本の都市|
|||個人的には京都の方がお得だと思います。|
|||知っている。あなたはエッフェル塔の頂上に行ったことがありますか?|
|Chinese |你对《三国演义》感兴趣吗?|是的,我。,喜欢各种人物,你怎么样?|
|||我很喜欢 三国啊。|
|||有一点喜欢,但是不是很了解,|
|||这.就是三国啊。|
|||嗯其实我对《三国演义》没什么兴趣因为就是一部!|
|||对啦,我喜欢各种人物!|
|||感兴趣吧?不过,我高中时候看的《三国演义》,虽然现在还是小学生,但是也喜欢玩杀,里面武将,人物!当然,最,以及故事,比如关羽,貂蝉。|
|||《三国志》是一部不错的史书,里面英雄人物塑造也很出色。值得一看!|
|||抖机灵《三国演义》是一部伟大的著作,里面有很多故事情节,历史上有名英雄人物,如关羽,|

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/GLM-Open-Dialogue](https://github.com/svjack/GLM-Open-Dialogue)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
-->
* [Bigscience](https://bigscience.huggingface.co)
* [FlagAI](https://github.com/FlagAI-Open/FlagAI)
* [TextBox](https://github.com/RUCAIBox/TextBox)
* [ClueAI](https://huggingface.co/ClueAI)
* [facebook](https://github.com/facebook)
* [DialoGPT](https://github.com/microsoft/DialoGPT)
* [EasyNMT](https://github.com/UKPLab/EasyNMT)
* [Stable-Diffusion-Chinese-Extend](https://github.com/svjack/Stable-Diffusion-Chinese-Extend)
* [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release)
* [prompt-extend](https://github.com/daspartho/prompt-extend)
* [svjack](https://huggingface.co/svjack)

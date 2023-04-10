<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">GLM-Open-Dialogue</h3>

  <p align="center">
   		一种由自回归空白填充任务通用语言预训练模型支持的开放式对话上下文生成器。
    <br />
  </p>
</p>

[In English](README_EN.md)

## 简要引述
### 引述及相关工作引述
根据开放领域对话生成的定义 [Open-domain Dialogue Generation:
What We Can Do, Cannot Do, And Should Do Next](https://aclanthology.org/2022.nlp4convai-1.13.pdf)

```
Given zero or more previous dialogue turns between
itself and one or more other participants,
a system  output a fluent, engaging,
and meaningful natural language response.
给出自身和一个或多个其他参与者之间的零个或多个先前对话轮次，
系统输出流畅、引人入胜且有意义的自然语言回复。
```

一个著名的处理此任务的模型是 [DialoGPT](https://github.com/microsoft/DialoGPT)，你可以在 [Roxza DialoGPT space](https://huggingface.co/spaces/Roxza/DialoGPT)上尝试它。

你也可以查看我的[svjack/Daliy-Dialogue](https://github.com/svjack/Daliy-Dialogue) 存储库，查看在这种任务上使用Bloom或GPT模型的效果。

### 由仓库 [svjack/Daliy-Dialogue](https://github.com/svjack/Daliy-Dialogue) 中给出的模型生成的数据集展示：

|名称 |HuggingFace 数据集链接| HuggingFace 空间链接 | 语言 |
|---------|--------|-------|-------|
| 英文每日对话模型生成例子 🦅🌸| https://huggingface.co/datasets/svjack/bloom-dialogue-generate-ds-en | https://huggingface.co/spaces/svjack/bloom-dialogue-english-sample-search| English |
| 中文对话模型生成例子 🐰🌸| https://huggingface.co/datasets/svjack/bloom-dialogue-generate-ds-zh | https://huggingface.co/spaces/svjack/bloom-gpt-dialogue-chinese-sample-search | Chinese |

### 这个工程的英文例子

下面是 DialoGPT and GLM 生成的对话上下文英文例子对比 （后者是此工程的工作）

|模型 |第一句话 | 生成的对话上下文 |
|---------|--------|-------|
|DialoGPT |Does money buy happiness? | Depends how much money you spend on it.|
|||  What is the best way to buy happiness ?|
|||  You just have to be a millionaire by your early 20s, then you can be happy .|
||| This is so difficult !|
|||You have no idea how hard it is to be a millionaire and happy . There is a reason the rich have a lot of money|
|GLM |Does money buy happiness? | Can money buy happiness?|
|||Yes, what about you?That is happiness.|
|||Oh yeah!|
|||You will be happy.|
|||Oh, I got it.How are you happy?|
|||I think no.|
|||Everyone has different definitions of happiness, and money cannot buy happiness.|
|DialoGPT |what is the meaning of life ? | The  meaning  is  to  be  with  your  family  and friends .|
|||I’m going to guess :  It means that your parents and friends have loved you . That’s the meaning , I believe .|
|||I think  the  meaning  is  to  make  sure  you’re alive when you die . So you’ll be with your family , friends , and yourself .|
|||It’s not , but thanks for trying to help .|
GLM |what is the meaning of life ? | You just like life, answer? |
|||of course is.|
|||Who said life itself?|
|||Does life have no meaning?|
|||Yes is.|
|||What do you like to do with me that is meaningful?|
|||Enjoy every moment of your life and feel meaningful.|

## HuggingFace 空间展示
* 1 多语言展示: https://huggingface.co/spaces/svjack/GLM-Open-Dialogue
* 2 中文展示: https://huggingface.co/spaces/svjack/GLM-Open-Dialogue-Chinese

## 安装和结构
```bash
pip install -r req0.txt
pip install -r req1.txt
```
### 结构
推荐使用gpu运行，如果有gpu,建议将下面文件
 [predict_t5_choose_simple.py](predict_t5_choose_simple.py) 的第一行从
```python
#### Run
#### pip install torch==1.8.0
device = "cpu"
```
变成
```python
#### Run
#### replace by your cuda version.
#### pip install torch==1.8.0+cu111
device = "cuda:0"
```
之后
```python
from main import *

#### 英文使用
##### max_times 从 2 改为 10 运行会花费更多时间，输出也更多
demo_func(*["Which keyboard do you prefer?", "en", 2])
demo_func(*["Which keyboard do you prefer?", "en", 10])
#### 日语使用
demo_func(*["東京と京都のどちらが住みやすいですか。", "ja", 2])
demo_func(*["東京と京都のどちらが住みやすいですか。", "ja", 10])

#### 中文使用
demo_func(*["程序员要掌握哪些技能", "zh", 5])
### 或者
generate_func(*["程序员要掌握哪些技能", 5])

generate_func(*["你对《三国演义》感兴趣吗?", 10])
generate_func(*["谁能让美国再一次伟大?", 10])
```

## 结构
### 从 GLM 到 开放领域对话生成的迁移

现有的一些模型通常在特定数据集上训练生成模型，这可能会受到数据集和预训练模型的限制。当面对广泛知识领域的开放域时，生成器可能会更容易出现“无意义问题”，即给出平淡无奇或缺乏生动感的回答。

该项目采用了一种无监督的组合方法来处理这些问题，并将其分解为四个步骤：

1 从头开始自行训练一个小数据集上的开放领域对话生成器，将其作为<b> “构建生成器”</b><br/>
2 使用[FlagAI](https://github.com/FlagAI-Open/FlagAI) 的 [GLM](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/glm_blank_filling/glm_generate_samples.py) 作为<b> “知识生成器”</b><br/>
3 使用自我训练的<b> “上下文判别器”</b> 来结合以上两个生成器。<br/>
4 使用自我训练的<b> “上下文重构器”</b> 来重构语境。<br/>

对于“构建生成器”，准确率要求不高。

以上四个模型已在[HuggingFace](https://huggingface.co)中发布，并在下表中列出。

|工程中的作用 |HuggingFace 链接| 贡献者 |
|---------|--------|-------|
|构建生成器| https://huggingface.co/svjack/T5-daliy-dialogue | https://huggingface.co/svjack |
|知识生成器| https://huggingface.co/BAAI/glm-large-chinese | https://huggingface.co/BAAI |
|上下文判别器| https://huggingface.co/svjack/T5-dialogue-choose | https://huggingface.co/svjack |
|上下文重构器| https://huggingface.co/svjack/T5-dialogue-collect | https://huggingface.co/svjack |

它们都由中文语料构建，并使用[facebook m2m_100_1.2B](https://huggingface.co/facebook/m2m100_1.2B)翻译模型使得它们支持多语言。

### Prompt --- 本工程贯穿的基本概念

这个工程有4个<b>Prompt</b>作为切入点，来构建这个工程。

|Prompt 种类|影响部分|概念来源|Prompt 的对应英文格式|Prompt 的对应中文格式|
|---------|--------|-------|--------|-------|
|面向填空问答的任务Prompt|[GLM](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/glm_blank_filling/glm_generate_samples.py)|build in GLM|Question: Is reading good for your grow up? Answer: [MASK]|问题: 阅读对于成长有好处吗？ 回答: [MASK]|
|面向选择最优的任务Prompt|[Context Discriminer](https://huggingface.co/svjack/T5-dialogue-choose)|[Bigscience-mt0](https://huggingface.co/bigscience/mt0-large)|According to the following context, choose the optimal follow-up sentence Context: {}\n{}Answer:|根据如下上下文，选择最优的后续句子 上下文：{}\n{}答案：|
|面向上下文重构成的任务Prompt|[Context Reconstructor](https://huggingface.co/svjack/T5-dialogue-collect)|[Bigscience-mt0](https://huggingface.co/bigscience/mt0-large)|According to the following context, reconstruct the Context: {}\n{}Answer:|根据下面的上下文进行分段： 上下文：{} 答案：|
|过滤低质量内容的类似Prompt概念|negative_prompts in [predict_t5_choose_simple.py](predict_t5_choose_simple.py)|[Stable Diffusion](https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/)|[Yes, I don't know, Who knows]|[是的,怎么知道]|



## 例子

|语言 |第一句话 | 生成的对话上下文 |
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
|Chinese |你对《三国演义》感兴趣吗?|是的,我喜欢各种人物。你觉得怎么样?|
|||我觉得可以,因为太无聊了。我喜欢《三国演义》,但是没时间去看。|
|||抖机灵,我喜欢历史,《三国志》。这些经典读起来不枯燥,有很强的吸引力,很适合初学者。|
|||哦,我也喜欢历史。你有兴趣吗?|
|||看历史,我感兴趣。|
|||毕竟,《三国志》是经典,但是要看清人物才能更好地理解人物性格。|

## 相似但更为学术化的类似其它工程作品
这篇论文名为“Controllable Factuality in Document-Grounded Dialog Systems Using a
Noisy Channel Model”，作者来自[UKPLab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/news_ukp/ukp_newsdetails_266432.en.jsp)。<br/>

论文链接: https://arxiv.org/pdf/2210.17418.pdf

Github 链接: https://github.com/ndaheim/noisy_channel_model

它也是关于对话系统生成和"修复"的一些类似问题，但主要与我这个项目的区别在于：<br/>

<b>
与我的“选择并重构”句子相比，它使用“解码和重构”标记。它使用 SentenceTransformer 来召回样本，而我使用 GLM 来“召回”知识。
</b>
<br/>

如果你对这个更加学术化的工作感兴趣，我建议你试一试。<br/>

## 对话生成的样本数据集

一个用于中文对话生成的样本数据集已经上传到 Huggingface Hub。[GLM-Open-Dialogue-Chinese-samples](https://huggingface.co/datasets/svjack/GLM-Open-Dialogue-Chinese-samples)<br/>

这个数据集中的问题是由抽取式问答构建的，在缺乏上下文的情况下回答起来有一些困难。如果你尝试其他问题（更难或更简单），输出结果可能会有所不同。

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - https://huggingface.co/svjack - svjackbt@gmail.com - ehangzhou@outlook.com

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

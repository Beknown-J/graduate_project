# Machine Learning Engineer Nanodegree
## Capstone Proposal
Jiang Ming 
March 21st, 2018

## Proposal
### Domain Background（项目背景）
​	本项目是要帮助Rossmann连锁药妆店的经理预测店铺未来的销售额，然而由于销售额会受到很多因素的影响，比如说折扣，节假日，季节，地理位置等因素。不同环境下的预测，其准确性会有所不同。因此，本项目选择坐落在德国的1115家店铺来预测其未来6个星期每天的销售额。准确的预测店铺未来一段时间的销售额，能够帮助经理更加合理的安排员工的工作时间，同时也有利于提升员工的工作效率。

​	这是一个标准的监督学习问题，因此可以使用常用的监督学习算法来预测销售额。最近几年比较火的XGBoost方法从基本的决策树方法发展而来，该方法屡次在各大数据比赛中获得第一[^1]。同时，该项目来自于kaggle，这个比赛的第一名使用的算法也是xgboost[^2]。因此，在本项目中，也准备尝试使用xgboost来解决这个项目的问题。之所以选择这个项目，是因为未来也想开始参加一些kaggle的比赛，这是一个好的开始。

### Problem Statement（问题描述）
​	本项目的数据集来自真实世界中Rossmann连锁药妆店的销售数据和店铺数据。一个很大的挑战在于，我们需要从现有的数据集中探索出店铺销售额的影响因素，从而才能预测店铺的销售额。另一个挑战在于，其预测周期为6周，较长的预测周期很有可能会给预测带来较大的误差，怎么才能找到影响店铺长期销售额的因素也很重要。总之，大量的特征工程和数据分析需要花费较多的时间。

​	寻找影响店铺销售额的影响因素需要从现有数据集的字段出发，比如说，可以从store.csv数据集获取更多店铺的信息，店铺的分类水平(Assortment)，促销活动(Promo)以及竞争对手店铺的距离(ComprtitionDistance)等都是影响销售额的影响因素。Gert也给出了一些建议，比如区分三种类型的特征：使用近期的数据构造特征；考虑时间信息；近期趋势。

### Datasets and Inputs（输入数据）
​	本项目的数据集由kaggle提供，使用的数据集主要由两个文件组成。一个是train.csv，是店铺的销售数据，主要字段比如销售额（Sales），客流量（Customers），日期（Date），促销信息（promo）等。另一个是store.csv，主要是1115家店铺的相关信息，主要字段有店铺类型（StoreType），竞争对手（Competitor），定期促销信息（Promotp2），摆放水平（Assortment）等。

​	要预测的是未来6周的销售额，这是时间序列的回归问题。可以把这个时间序列数据转化为横截面数据，因此在构造一些特征时，就要考虑到特征与时间的相关性。考虑把train.csv的数据划分为3份，第一份从2013-01-01到2015-05-08的数据作为train data，第二份从2015-05-09到2015-06-19的数据作为validation data，第三份从2015-06-20到2015-07-31的数据作为test data。数据划分时要按照时间来划分。

### Solution Statement（解决办法）
​	本项目选择xgboost模型来解决该问题，xgboost是以CART作为基分类器的。CART能够解决回归问题，并且决策树算法能够给出每个特征的重要程度，有利于模型的特征筛选。同时，xgboost模型借鉴了随机森林的思想，支持列抽样，不仅能够防止过拟合，还能减少计算量。本项目的一个难点就在于特征的构造，然而可以利用xgboost的特点，来找出影响销售额的重要变量。

### Benchmark Model（基准模型）

​	在kaggle上，Gert在本项目上获得了第一名的成绩。以RMSPE为评价指标，其得分为0.10021。Gert[^2]使用的就是集成的xgboost 模型，也就是多个采用不同特征的xgboost模型集成。该比赛获得金奖的指标最低得分是0.11037，获得银奖的指标最低得分是0.11552，获得铜奖的最低得分是0.11773。由于能够参考很多前人的研究，特别是Gert给出了自己获得第一名的比赛文档，本项目设定的目标为RMSPE指标得分在0.11552以内。

### Evaluation Metrics（评估指标）
​	本项目的评价指标由kaggle给出， Root Mean Square Percentage Error (RMSPE)，而且有比赛的结果作为对比。因此，选择RMSPE作为评价指标，定义为：

$$RMSPE=\sqrt{\frac{1}{n}\sum_{i=1}^N\frac{y_i-\hat{y_i}}{y_i}}$$ 

其中$y_i$表示真实的销售额，$\hat{y_i}$表示销售额的预测值。当销售额为0时，不带入计算。

### Project Design（设计大纲）
​	本项目主要分以下几个步骤：数据预处理，数据分析，特征工程，模型训练，模型预测和评价。下面分别介绍每个流程的工作。

​	数据预处理。通过观察数据发现，在store.csv文件中，i）有一些字段下的数据为空值，考虑使用0来填充； ii）对有些特征进行one-hot-vect处理，DayOfWeek，Assortment，StoreType，PromolInterval都是有限的类别，而且是字符型数据，所以要进行one-hot-vect处理，但是这么处理可能会使数据比较稀疏，特别是对PromolInterval中值的类别可能会比较多，但这是一种尝试；iii）考虑是否删掉销售额为0的数据，因为在评价时，当真实销售额为0时，是不带入计算的。训练时，考虑是否保留这些数据。

​	数据分析。要预测的目标是未来6周的销售额，那么销售额是否会存在随时间变化的趋势。如果存在，周期大概是多少。通过下图来观察店铺1的销售额随时间变化的趋势。![图1](./ana1.png "Store 1") 

​	从上图的销售额随时间变化的趋势可以看出来，i）每个月销售额会有相似的变化趋势，比如2013-01-08到2013-02-05的销售额和2013-02-05到2013-03-05的销售额，会呈现相似的波动趋势。因此，未来一个月的销售额，与前一个月有很强的相关关系，在构造特征时，需要注意这一点。ii）值得注意的是，销售额与DayOfWeek有很强的相关性，因为图中很明显的可以看出，周日是不营业的，销售额为0。iii）其余店铺均有同节结论，这里只展示了店铺1。类似于上述分析，还需要进行更多的深入思考，更详细的分析在毕业设计里展示。通过分析发现特征与销售额之间存在的隐含关系，有利于构造更多的特征。

​	特征工程。现有的数据中，有很多字段（特征）是对销售额有重要的影响。比如，DayOfWeek，Customers，Promo，SchoolHoliday，Competitor，Promo2SinceWeek等。除了这些特征以外，根据Gert[^2]的建议，构造更多的特征。对于特征，Gert指出了三种类型的特征。第一种是Recent data，第二种是Terporal information，第三种是Current data，以及其他信息。在毕业设计中，我会根据Gert的建议来设计特征工程，并且解释，为什么提取上述特征。

​	模型训练。在构造完特征后，要带入xgboost模型训练。参考Gert[^2]的做法，对于训练不同的xgboost模型，并且将训练结果融合在一起，观察是否能提升模型的预测能力。对与不同的模型，其实是指使用不同特征的训练的xgboost模型。

​	模型预测和评价。在test data上预测销售额，并使用RMSPE指标评价模型的预测效果。

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?

### Reference

[^1]: Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System[C]// ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016:785-794.
[^2]: [Gert's model doucmentation.](https://kaggle2.blob.core.windows.net/forum-message-attachments/102102/3454/Rossmann_nr1_doc.pdf) 


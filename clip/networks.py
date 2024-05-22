import torch
import torch.nn as nn
import torchvision.models as models


class FusionModel(nn.Module):
    def __init__(self, num_classes_img, num_classes_gender=None, num_classes_age=None):
        super(FusionModel, self).__init__()
        
        # 图像特征提取器（DenseNet）
        # self.base_model = models.densenet161(pretrained=False, num_classes=num_classes_img)
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(in_features=2048, out_features=1280, bias=True)
        
        # 处理性别信息的部分
        self.gender_module = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # 处理年龄信息的部分
        self.age_module = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 处理年龄信息的部分
        self.gene_module = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 处理年龄信息的部分
        self.asp_module = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 处理年龄信息的部分
        self.tirads_module = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 处理性别信息的部分
        self.def_module = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # 处理年龄信息的部分
        self.border_module = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 处理年龄信息的部分
        self.echo_module = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 处理年龄信息的部分
        self.cal_module = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 处理年龄信息的部分
        self.pro_module = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 最终分类器
        self.final_classifier = nn.Linear(1280 + 128*2, num_classes_img)
        # self.final_classifier = nn.Linear(1280 + 128*10, num_classes_img)
        # self.final_classifier = nn.Linear(128*10, num_classes_img)

    def forward(self, img, age_info, gender_info): # gene_info, asp_info, def_info, border_info, echo_info, cal_info, pro_info,  tirads_info
        # return self.base_model(img)
        # 提取图像特征
        img_features = self.base_model(img)
        
        gender_features = self.gender_module(gender_info)
        age_features = self.age_module(age_info)
        # gene_features = self.gene_module(gene_info)
        # asp_features = self.asp_module(asp_info)
        # def_features = self.def_module(def_info)
        # border_features = self.border_module(border_info)
        # echo_features = self.echo_module(echo_info)
        # cal_features = self.cal_module(cal_info)
        # pro_features = self.pro_module(pro_info)
        # tirads_features = self.tirads_module(tirads_info)

        # 合并图像特征、性别信息和年龄信息
        combined_features = torch.cat((img_features, gender_features, age_features), dim=1)
        # 合并图像特征、性别信息和年龄信息
        # combined_features = torch.cat((img_features, age_features, gender_features, gene_features, asp_features, tirads_features), dim=1)
        # combined_features = torch.cat((age_features, gender_features), dim=1)
        # combined_features = torch.cat((img_features, age_features, gender_features, gene_features, asp_features, tirads_features, \
        #     def_features, border_features, echo_features, cal_features, pro_features), dim=1)
        # combined_features = torch.cat((age_features, gender_features, gene_features, asp_features, tirads_features, \
        #     def_features, border_features, echo_features, cal_features, pro_features), dim=1)
        # 最终分类
        output = self.final_classifier(combined_features)

        return output

# # 设置属性信息的维度和类别数
# num_gender_attributes = 2
# num_age_attributes = 1
# num_classes_img = 3
# num_classes_gender = 2
# num_classes_age = 5

# # 创建模型
# custom_model = MyModel(num_classes_img, num_classes_gender, num_classes_age)

# # 输入示例
# img_input = torch.randn(1, 3, 224, 224)  # 图像输入
# gender_input = torch.randn(1, num_gender_attributes)  # 性别信息输入
# age_input = torch.randn(1, num_age_attributes)  # 年龄信息输入

# # 前向传播
# output = custom_model(img_input, gender_input, age_input)
# print(output.shape)

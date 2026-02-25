import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import json
import os
import warnings
import re

# ===================== 页面配置（必须第一行） =====================
st.set_page_config(
    page_title="皮肤病智能诊断系统（多模态）",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)
warnings.filterwarnings("ignore")

# ===================== 核心配置（完全沿用你高准确率版本） =====================
MODEL_PATH = "./models/best_mnv3_macroF1.pth"
LABEL_PATH = "./label_map.json"
INPUT_SIZE = 320
NUM_CLASSES = 31

# ===================== 严格匹配你的 label_map.json =====================
id2label = {
    0: "Acne Vulgaris",
    1: "Actinic Keratosis",
    2: "Alopecia Areata",
    3: "Arsenic",
    4: "Basal Cell Carcinoma",
    5: "Chickenpox",
    6: "Contact Dermatitis",
    7: "Cowpox",
    8: "Dermatofibroma",
    9: "Eczema",
    10: "Exanthems",
    11: "Folliculitis",
    12: "HFMD",
    13: "Healthy",
    14: "Herpes Simplex",
    15: "Herpes Zoster",
    16: "Impetigo",
    17: "Lichen Planus",
    18: "Lichenoid Keratosis",
    19: "Measles",
    20: "Melanocytic Nevus",
    21: "Melanoma",
    22: "Monkeypox",
    23: "Onychomycosis",
    24: "Pityriasis Rosea",
    25: "Psoriasis",
    26: "Scabies",
    27: "Seborrheic Keratosis",
    28: "Squamous Cell Carcinoma",
    29: "Urticaria",
    30: "Vascular Lesion"
}

# 中文标签映射
cn_label = {
    "Acne Vulgaris": "寻常痤疮",
    "Actinic Keratosis": "光化性角化病",
    "Alopecia Areata": "斑秃",
    "Arsenic": "砷剂皮肤病",
    "Basal Cell Carcinoma": "基底细胞癌",
    "Chickenpox": "水痘",
    "Contact Dermatitis": "接触性皮炎",
    "Cowpox": "牛痘",
    "Dermatofibroma": "皮肤纤维瘤",
    "Eczema": "湿疹",
    "Exanthems": "发疹病",
    "Folliculitis": "毛囊炎",
    "HFMD": "手足口病",
    "Healthy": "健康皮肤",
    "Herpes Simplex": "单纯疱疹",
    "Herpes Zoster": "带状疱疹",
    "Impetigo": "脓疱疮",
    "Lichen Planus": "扁平苔藓",
    "Lichenoid Keratosis": "苔藓样角化病",
    "Measles": "麻疹",
    "Melanocytic Nevus": "黑素细胞痣",
    "Melanoma": "黑色素瘤",
    "Monkeypox": "猴痘",
    "Onychomycosis": "甲癣",
    "Pityriasis Rosea": "玫瑰糠疹",
    "Psoriasis": "银屑病",
    "Scabies": "疥疮",
    "Seborrheic Keratosis": "脂溢性角化病",
    "Squamous Cell Carcinoma": "鳞状细胞癌",
    "Urticaria": "荨麻疹",
    "Vascular Lesion": "血管性病变"
}

# ===================== 疾病文本库（用于文本相似度匹配） =====================
disease_text = {
    "Acne Vulgaris": "面部及背部出现散在的粉刺、炎性丘疹及脓疱。伴有明显的皮脂溢出，皮肤外观油腻，毛孔粗大。可见黑头粉刺（开口）及白头粉刺（闭口），伴红肿。重症表现为深在的结节、囊肿或聚合性斑块，触痛明显。皮损愈合后遗留暂时性红斑、色素沉着或凹陷性瘢痕。",
    "Actinic Keratosis": "曝光部位出现的干燥、粗糙、附着性鳞屑性红斑。触感如砂纸般粗糙，基底呈淡红色或暗红色。表面有角质增生，强行剥离鳞屑可见出血点。长期日晒后的癌前病变，多发于面部、头皮或手背。黄褐色或黑褐色角化性皮损，周围皮肤有光老化表现。",
    "Alopecia Areata": "突然发生的圆形或椭圆形脱发区，边界清楚。脱发区头皮光滑，无红肿、脱屑或萎缩。边缘的头发松动易拔，发根呈感叹号状（上粗下细）。可单发或多发，严重者全头皮头发脱落（全秃）。指甲可出现点状凹陷或纵嵴。",
    "Arsenic": "砷剂皮肤病",
    "Basal Cell Carcinoma": "珍珠样隆起的结节，表面可见扩张的毛细血管（红血丝）。边缘呈卷曲状的溃疡（鼠咬状溃疡），中央凹陷，质地较硬。蜡样光泽的半透明丘疹，缓慢生长，易破溃出血。表面覆盖结痂的红斑或斑块，去除结痂后易出血。常见于面部曝光区，呈黑色或褐色的色素性基底细胞癌。",
    "Chickenpox": "水痘",
    "Contact Dermatitis": "接触部位出现境界清楚的红斑、水肿、丘疹。严重时可见水疱、大疱，破溃后有糜烂和渗液。皮损形状与接触物（如手表、皮带）一致，去除病因后消退。伴有烧灼感或剧烈瘙痒，搔抓后可扩散。慢性期表现为皮肤轻度浸润、肥厚或苔藓样变。",
    "Cowpox": "牛痘",
    "Dermatofibroma": "坚实的圆形结节，表面呈红褐色或黄褐色。捏起皮损两侧，表面出现凹陷（“酒窝征”阳性）。质地坚硬如扣子，在皮肤内可自由移动，无明显压痛。表面皮肤光滑或轻度角化，多发于四肢伸侧。虫咬或外伤后出现的缓慢生长的纤维组织增生。",
    "Eczema": "局部皮肤出现多形性皮疹，红斑基础上见丘疹，伴剧烈瘙痒。急性期可见红斑、水肿，抓破后有明显的液体渗出倾向。皮肤干燥、粗糙、增厚，伴有抓痕、血痂及色素沉着。皮损呈对称性分布，反复发作，易演变为慢性苔藓样变。患处皮肤纹理加深，呈皮革样外观，边界模糊不清。",
    "Exanthems": "全身弥漫性分布的鲜红色斑疹或斑丘疹（麻疹样）。发病突然，常有用药史或病毒感染前驱症状。皮损对称分布，压之褪色，伴有瘙痒。严重者可伴有黏膜损害（口腔溃疡）或发热。停药或病毒感染好转后，皮损逐渐消退，可有脱屑。",
    "Folliculitis": "以毛囊为中心的红色丘疹，顶端有白色脓疱。周围有红晕，伴有轻度疼痛或瘙痒。脓疱破溃后排出少量脓液，结痂愈合，不留瘢痕。多发于头皮、面部、颈部或背部多毛区。反复发作可形成疖肿，局部红肿热痛。",
    "HFMD": "手足口病",
    "Healthy": "健康皮肤",
    "Herpes Simplex": "皮肤黏膜交界处（如口角、唇缘）出现群集性小水疱。局部有灼热、瘙痒或刺痛感。水疱易破溃，形成浅表糜烂或结痂，易反复发作。常由感冒、发热、疲劳或日晒诱发。典型的“上火”表现，病程约1周左右。",
    "Herpes Zoster": "沿单侧神经分布的簇集性水疱，不超过身体中线。水疱基底发红，疱壁紧张发亮，疱液澄清。伴有明显的神经痛（烧灼感、针刺感或电击感）。发疹前可有发热、乏力或局部皮肤感觉过敏。愈合后可遗留暂时性红斑、色素沉着或后遗神经痛。",
    "Impetigo": "红色斑点或丘疹迅速转变为脓疱，周围有红晕。脓疱破裂后流出黄水，干燥后形成蜜黄色厚痂。多见于儿童面部口周或鼻孔周围，有接触传染性。伴有瘙痒，搔抓后接种传播到身体其他部位。严重者可出现大疱，疱液浑浊，全身症状较轻。",
    "Lichen Planus": "紫红色多角形扁平丘疹，表面有蜡样光泽。表面可见灰白色网状纹理（Wickham纹）。剧烈瘙痒，多发于手腕、前臂屈侧或脚踝。愈合后遗留明显的色素沉着。搔抓后可在抓痕处出现新皮损（同形反应）。",
    "Lichenoid Keratosis": "苔藓样角化病",
    "Measles": "麻疹",
    "Melanocytic Nevus": "边界清晰、形状规则的圆形或椭圆形色素斑，颜色均匀。表面平滑或轻度隆起，颜色呈均匀的棕色或黑褐色。长期稳定存在的色素性皮损，无明显增大或颜色改变。典型的良性黑素细胞增生，结构对称，边缘整齐。均质的色素沉着区域，表面纹理正常，无溃疡或出血。",
    "Melanoma": "黑色或深褐色不对称斑块，边缘呈现不规则锯齿状，界限不清。皮损颜色不均匀，可见深浅不一的色素沉着（黑、褐、红白相间）。病灶直径通常大于6mm，且近期有明显增大、隆起或形状改变。表面可能出现溃疡、出血、结痂，或伴有局部瘙痒、疼痛感。形状不对称，边界模糊不清的色素性皮损，周围可见卫星灶。",
    "Monkeypox": "猴痘",
    "Onychomycosis": "指（趾）甲板变浑浊、无光泽，呈黄色或灰褐色。甲板增厚、变脆，表面凹凸不平或破损。甲下有角蛋白碎屑堆积，甲板与甲床分离。甲板变形、萎缩或脱落，严重影响美观。常由足癣或手癣传染而来，进展缓慢。",
    "Pityriasis Rosea": "躯干或四肢近端出现椭圆形玫瑰色红斑，表面有糠状鳞屑。先出现一个较大的母斑，随后出现继发性子斑。皮损长轴与皮纹走向（肋骨方向）一致，呈“圣诞树”样分布。边缘有领圈样脱屑，中心消退。伴有轻度瘙痒，具有自限性，一般不复发。",
    "Psoriasis": "境界清楚的红斑，表面覆盖多层银白色鳞屑。刮除鳞屑可见薄膜现象，继续刮除可见点状出血（Auspitz征）。头部可见束状发，或指甲出现顶针样凹陷（点状凹陷）。皮损多位于四肢伸侧（如肘、膝关节），呈钱币状或地图状。伴有不同程度的瘙痒，冬季加重，夏季缓解。",
    "Scabies": "指缝、腕屈侧等皮肤薄嫩处出现针头大丘疹、丘疱疹。可见灰白色或浅黑色的线状隧道（疥虫挖掘）。夜间瘙痒剧烈，严重影响睡眠。男性阴囊或阴茎上可见红褐色结节（疥疮结节）。家庭成员或集体生活中常有类似症状者。",
    "Seborrheic Keratosis": "边界清楚的褐色或黑色斑块，表面呈“贴在皮肤上”的样貌。表面粗糙，可见油脂性鳞屑或痂皮，剥离后基底不出血。呈乳头瘤样增生，表面有裂隙，像“蜡滴”滴在皮肤上。老年常见的良性表皮增生，颜色深浅不一，质地柔软。表面可见假性角囊肿（黑头粉刺样小点），无自觉症状。",
    "Squamous Cell Carcinoma": "坚硬的红色结节或斑块，表面有厚角质层或鳞屑附着。生长迅速的溃疡性肿物，边缘隆起，底部凹凸不平。菜花状或疣状增生，质地脆，触之易出血。基底浸润感明显，可伴有疼痛或压痛。长期不愈合的角化性皮损，表面破溃并有脓性分泌物。",
    "Urticaria": "皮肤突然出现大小不等的风团（风疙瘩），呈鲜红或苍白色。风团周围有红晕，伴有剧烈瘙痒或烧灼感。皮损发作迅速，消退后不留痕迹，24小时内消退。搔抓后皮肤出现条状隆起（皮肤划痕征阳性）。可伴有血管性水肿（嘴唇或眼睑肿胀）。",
    "Vascular Lesion": "鲜红色或紫红色的斑块/结节，压之褪色或部分褪色。柔软的草莓状肿物，高出皮面，颜色鲜红。散在的樱桃红样小丘疹，不痛不痒，随年龄增多。紫红色的葡萄酒色斑，边界清楚，不高出皮面。皮肤表面可见放射状扩张的毛细血管，呈蜘蛛网状。"
}

# ===================== 模型加载（完全保留你原来的高准确率模型） =====================
@st.cache_resource
def load_model():
    try:
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model, True
    except Exception as e:
        return None, False

model, model_ok = load_model()

# ===================== 图片预处理（完全不变） =====================
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# ===================== 文本相似度（轻量，不影响图片精度） =====================
def text_similarity(text, user_input):
    if not user_input:
        return 0.0
    # 简单的词频匹配
    user_words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', user_input))
    text_words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', text))
    if not user_words:
        return 0.0
    return len(user_words & text_words) / len(user_words)

# ===================== 三合一推理：图片 / 文本 / 融合 =====================
def predict(img, user_text="", mode="image"):
    # 1. 图像得分（完全用你原来的模型，保证最高准确率）
    img_score = torch.zeros(1, NUM_CLASSES)
    if mode in ["image", "fusion"] and img is not None:
        x = preprocess(img)
        with torch.no_grad():
            out = model(x)
            img_score = torch.softmax(out, dim=1)

    # 2. 文本得分（轻量匹配）
    text_score_tensor = torch.zeros(1, NUM_CLASSES)
    if mode in ["text", "fusion"] and user_text.strip():
        scores = []
        for i in range(NUM_CLASSES):
            cls_name = id2label.get(i, "")
            desc = disease_text.get(cls_name, "")
            scores.append(text_similarity(desc, user_text))
        text_score_tensor = torch.tensor(scores).unsqueeze(0)
        text_score_tensor = torch.softmax(text_score_tensor * 8, dim=1)  # 放大置信度

    # 3. 融合（图像占70%，保证准确率）
    if mode == "image":
        final = img_score
    elif mode == "text":
        final = text_score_tensor
    else:
        final = 0.7 * img_score + 0.3 * text_score_tensor

    top3_prob, top3_idx = torch.topk(final, 3)
    res = []
    for i in range(3):
        c = top3_prob[0][i].item() * 100
        idx = top3_idx[0][i].item()
        cls = id2label.get(idx, "未知")
        res.append((cn_label.get(cls, cls), c))
    return res

# ===================== 界面（和你原来一样 + 加文本框） =====================
st.title("🩺 皮肤病智能诊断系统（多模态融合）")
st.markdown("---")

if model_ok:
    st.success("✅ 高准确率图像模型加载成功！")
else:
    st.error("❌ 模型加载失败")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 上传图片")
    img_file = st.file_uploader("JPG / PNG", type=["jpg","jpeg","png"])
    img = None
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)
        st.success("✅ 图片已上传")

with col2:
    st.subheader("📝 症状描述（可选）")
    text = st.text_area("填写症状（不填则只使用图片）", height=120,
                        placeholder="例如：红色斑块、瘙痒、脱屑、边界不规则...")
    st.subheader("🔧 诊断模式")
    mode = st.radio("", ["仅图片（最高准确率）", "仅文本", "图片+文本融合"], horizontal=True)
    mode_map = {
        "仅图片（最高准确率）": "image",
        "仅文本": "text",
        "图片+文本融合": "fusion"
    }
    go = st.button("🚀 开始诊断", type="primary", use_container_width=True)

# ===================== 结果展示 =====================
st.markdown("---")
st.subheader("🏥 诊断结果")

if go and model_ok:
    if mode in ["image","fusion"] and img is None:
        st.warning("⚠️ 请先上传图片")
    elif mode in ["text","fusion"] and not text.strip():
        st.warning("⚠️ 请输入症状描述")
    else:
        with st.spinner("推理中..."):
            res = predict(img, text, mode_map[mode])
        st.success("✅ 诊断完成")
        for name, conf in res:
            st.write(f"• **{name}** — {conf:.1f}%")

        st.warning("⚠️ 本结果仅供辅助参考，不替代医师诊断")

elif img is None:
    st.info("💡 请上传图片开始诊断")
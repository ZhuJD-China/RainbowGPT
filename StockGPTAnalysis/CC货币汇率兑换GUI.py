import tkinter as tk
from tkinter import ttk
import requests
import json
from decimal import Decimal, ROUND_HALF_UP

# 从fixer.io API获取汇率
url = "http://data.fixer.io/api/latest?access_key=33ec7c73f8a4eb6b9b5b5f95118b2275"
data = requests.get(url).text
data2 = json.loads(data)
fx = data2["rates"]


class CurrencyConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("货币转换器")

        self.label_amount = ttk.Label(root, text="金额:")
        self.label_amount.grid(row=0, column=0, padx=10, pady=10)

        self.entry_amount = ttk.Entry(root)
        self.entry_amount.grid(row=0, column=1, padx=10, pady=10)

        self.label_from = ttk.Label(root, text="源货币:")
        self.label_from.grid(row=1, column=0, padx=10, pady=10)

        self.combo_from = ttk.Combobox(root, values=list(fx.keys()))
        self.combo_from.grid(row=1, column=1, padx=10, pady=10)

        self.label_to = ttk.Label(root, text="目标货币:")
        self.label_to.grid(row=2, column=0, padx=10, pady=10)

        self.combo_to = ttk.Combobox(root, values=list(fx.keys()))
        self.combo_to.grid(row=2, column=1, padx=10, pady=10)

        self.label_result = ttk.Label(root, text="")
        self.label_result.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        self.convert_button = ttk.Button(root, text="转换", command=self.convert)
        self.convert_button.grid(row=4, column=0, columnspan=2, pady=10)

    def convert(self):
        try:
            amount = Decimal(self.entry_amount.get())
            from_currency = self.combo_from.get().upper()
            to_currency = self.combo_to.get().upper()

            result = amount * Decimal(fx[to_currency]) / Decimal(fx[from_currency])
            result = result.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)

            self.label_result.config(text=f"{amount} {from_currency} 相当于 {result} {to_currency}（今天）")
        except ValueError:
            self.label_result.config(text="无效的输入，请输入有效的数字。")


if __name__ == "__main__":
    root = tk.Tk()
    converter = CurrencyConverter(root)
    root.mainloop()

"""
    "AED : 阿联酋迪拉姆，阿拉伯联合酋长国迪拉姆",
    "AFN : 阿富汗阿富汗尼，阿富汗阿富汗尼",
    "ALL : 阿尔巴尼亚列克，阿尔巴尼亚列克",
    "AMD : 亚美尼亚德拉姆，亚美尼亚德拉姆",
    "ANG : 荷兰盾，荷属安的列斯盾，博奈尔，库拉索，萨巴，圣尤斯特歇斯，圣马丁",
    "AOA : 安哥拉宽扎，安哥拉宽扎",
    "ARS : 阿根廷比索，阿根廷比索，马尔维纳斯群岛",
    "AUD : 澳大利亚元，澳大利亚元，圣诞岛，科科斯（基林）群岛，诺福克岛，阿什莫尔和卡特尔群岛，澳大利亚南极领地，珊瑚海岛，赫德岛，麦克唐纳群岛，基里巴斯，瑙鲁",
    "AWG : 阿鲁巴盾，阿鲁巴盾",
    "AZN : 阿塞拜疆马纳特，阿塞拜疆马纳特",
    "BAM : 波斯尼亚可兑换马克，波斯尼亚和黑塞哥维那可兑换马克",
    "BBD : 巴巴多斯元，巴巴多斯元",
    "BDT : 孟加拉塔卡，孟加拉国塔卡",
    "BGN : 保加利亚列弗，保加利亚列弗",
    "BHD : 巴林第纳尔，巴林第纳尔",
    "BIF : 布隆迪法郎，布隆迪法郎",
    "BMD : 百慕大元，百慕大元",
    "BND : 文莱元，文莱元",
    "BOB : 玻利维亚诺，玻利维亚诺",
    "BRL : 巴西雷亚尔，巴西雷亚尔",
    "BSD : 巴哈马元，巴哈马元",
    "BTC : 比特币，比特币，XBT",
    "BTN : 不丹努尔特鲁姆，不丹努尔特鲁姆",
    "BWP : 博茨瓦纳普拉，博茨瓦纳普拉",
    "BYN : 白俄罗斯卢布，白俄罗斯卢布",
    "BYR : 白俄罗斯卢布，白俄罗斯卢布",
    "BZD : 伯利兹元，伯利兹元",
    "CAD : 加拿大元，加拿大元",
    "CDF : 刚果法郎，刚果（金）法郎",
    "CHF : 瑞士法郎，瑞士法郎，列支敦士登，坎皮奥内迪塔利亚，比辛根安阔因",
    "CLF : 智利第一账户",
    "CLP : 智利比索，智利比索",
    "CNY : 人民币，人民币",
    "COP : 哥伦比亚比索，哥伦比亚比索",
    "CRC : 哥斯达黎加科朗，哥斯达黎加科朗",
    "CUC : 古巴可兑换比索，古巴可兑换比索",
    "CUP : 古巴比索，古巴比索",
    "CVE : 佛得角埃斯库多，佛得角埃斯库多",
    "CZK : 捷克克朗，捷克克朗",
    "DJF : 吉布提法郎，吉布提法郎",
    "DKK : 丹麦克朗，丹麦克朗，法罗群岛，格陵兰",
    "DOP : 多米尼加比索，多米尼加共和国比索",
    "DZD : 阿尔及利亚第纳尔，阿尔及利亚第纳尔",
    "EGP : 埃及镑，埃及镑，加沙地带",
    "ERN : 厄立特里亚纳克法，厄立特里亚纳克法",
    "ETB : 埃塞俄比亚比尔，埃塞俄比亚比尔，厄立特里亚",
    "EUR : 欧元，欧元成员国，安道尔，奥地利，亚速尔群岛，巴利阿里群岛，比利时，加那利群岛，塞浦路斯，芬兰，法国，法属圭亚那，法属南部领地，德国，希腊，瓜德罗普，荷兰（荷兰），梵蒂冈城，爱尔兰，意大利，卢森堡，马德拉群岛，马耳他，摩纳哥，黑山，荷兰",
    "FJD : 斐济元，斐济元",
    "FKP : 福克兰群岛镑，福克兰群岛（马尔维纳斯群岛）镑",
    "GBP : 英镑，英国镑，英国（英国），英格兰，北爱尔兰，苏格兰，威尔士，福克兰群岛，直布罗陀，根西，马恩岛，泽西，圣赫勒拿和阿森松，南乔治亚和南桑威奇群岛，特里斯坦达昆哈",
    "GEL : 格鲁吉亚拉里，格鲁吉亚拉里",
    "GGP : 根西镑，根西镑",
    "GHS : 加纳塞地，加纳塞地",
    "GIP : 直布罗陀镑，直布罗陀镑",
    "GMD : 冈比亚达拉西，冈比亚达拉西",
    "GNF : 几内亚法郎，几内亚法郎",
    "GTQ : 危地马拉格查尔，危地马拉格查尔",
    "GYD : 圭亚那元，圭亚那元",
    "HKD : 港元，港元",
    "HNL : 洪都拉斯伦皮拉，洪都拉斯伦皮拉",
    "HRK : 克罗地亚库纳，克罗地亚库纳",
    "HTG : 海地古德，海地古德",
    "HUF : 匈牙利福林，匈牙利福林",
    "IDR : 印度尼西亚盾，印度尼西亚盾，东帝汶",
    "ILS : 以色列新谢克尔，以色列谢克尔，巴勒斯坦领土",
    "IMP : 英属曼岛镑，英属曼岛镑",
    "INR : 印度卢比，印度卢比，不丹，尼泊尔",
    "IQD : 伊拉克第纳尔，伊拉克第纳尔",
    "IRR : 伊朗里亚尔，伊朗里亚尔",
    "ISK : 冰岛克朗，冰岛克朗",
    "JEP : 泽西岛镑，泽西岛镑",
    "JMD : 牙买加元，牙买加元",
    "JOD : 约旦第纳尔，约旦第纳尔",
    "JPY : 日元，日本日元",
    "KES : 肯尼亚先令，肯尼亚先令",
    "KGS : 吉尔吉斯斯坦索姆，吉尔吉斯斯坦索姆",
    "KHR : 柬埔寨瑞尔，柬埔寨瑞尔",
    "KMF : 科摩罗法郎，科摩罗法郎",
    "KPW : 朝鲜元，朝鲜元",
    "KRW : 韩元，韩元",
    "KWD : 科威特第纳尔，科威特第纳尔",
    "KYD : 开曼元，开曼群岛元",
    "KZT : 哈萨克斯坦坚戈，哈萨克斯坦坚戈",
    "LAK : 老挝基普，老挝基普",
    "LBP : 黎巴嫩镑，黎巴嫩镑",
    "LKR : 斯里兰卡卢比，斯里兰卡卢比",
    "LRD : 利比里亚元，利比里亚元",
    "LSL : 莱索托洛蒂，莱索托洛蒂",
    "LTL : 立陶宛立特",
    "LVL : 拉脱维亚拉特",
    "LYD : 利比亚第纳尔，利比亚第纳尔",
    "MAD : 摩洛哥迪拉姆，摩洛哥迪拉姆，西撒哈拉",
    "MDL : 摩尔多瓦列伊，摩尔多瓦列伊",
    "MGA : 马达加斯加阿里亚里，马达加斯加阿里亚里",
    "MKD : 马其顿第纳尔，马其顿第纳尔",
    "MMK : 缅甸缅元，缅甸缅元",
    "MNT : 蒙古图格里克，蒙古图格里克",
    "MOP : 澳门澳门元，澳门澳门元",
    "MRU : 毛里塔尼亚乌吉亚，毛里塔尼亚乌吉亚",
    "MUR : 毛里求斯卢比，毛里求斯卢比",
    "MVR : 马尔代夫拉菲亚，马尔代夫拉菲亚",
    "MWK : 马拉维克瓦查，马拉维克瓦查",
    "MXN : 墨西哥比索，墨西哥比索",
    "MYR : 马来西亚林吉特，马来西亚林吉特",
    "MZN : 莫桑比克梅蒂卡尔，莫桑比克梅蒂卡尔",
    "NAD : 纳米比亚元，纳米比亚元",
    "NGN : 尼日利亚奈拉，尼日利亚奈拉",
    "NIO : 尼加拉瓜科多巴，尼加拉瓜科多巴",
    "NOK : 挪威克朗，挪威克朗，布维岛，斯瓦尔巴，扬·迈恩，莫德女王地，彼得一世岛",
    "NPR : 尼泊尔卢比，尼泊尔卢比，印度（非官方，靠近印度尼泊尔边界）",
    "NZD : 新西兰元，新西兰元，库克群岛，纽埃，皮特凯恩群岛，托克劳",
    "OMR : 阿曼里亚尔，阿曼里亚尔",
    "PAB : 巴拿马巴尔博亚，巴拿马巴尔博亚",
    "PEN : 秘鲁索尔，秘鲁索尔",
    "PGK : 巴布亚新几内亚基那，巴布亚新几内亚基那",
    "PHP : 菲律宾比索，菲律宾比索",
    "PKR : 巴基斯坦卢比，巴基斯坦卢比",
    "PLN : 波兰兹罗提，波兰兹罗提",
    "PYG : 巴拉圭瓜拉尼，巴拉圭瓜拉尼",
    "QAR : 卡塔尔里亚尔，卡塔尔里亚尔",
    "RON : 罗马尼亚列伊，罗马尼亚列伊",
    "RSD : 塞尔维亚第纳尔，塞尔维亚第纳尔",
    "RUB : 俄罗斯卢布，俄罗斯卢布，塔吉克斯坦，阿布哈兹，南奥塞梯",
    "RWF : 卢旺达法郎，卢旺达法郎",
    "SAR : 沙特阿拉伯里亚尔，沙特阿拉伯里亚尔",
    "SBD : 所罗门群岛元，所罗门群岛元",
    "SCR : 塞舌尔卢比，塞舌尔卢比",
    "SDG : 苏丹镑，苏丹镑",
    "SEK : 瑞典克朗，瑞典克朗",
    "SGD : 新加坡元，新加坡元",
    "SHP : 圣赫勒拿岛镑，圣赫勒拿岛镑",
    "SLL : 塞拉利昂利昂，塞拉利昂利昂",
    "SOS : 索马里先令，索马里先令",
    "SRD : 苏里南元，苏里南元",
    "STN : 圣多美和普林西比多布拉，圣多美和普林西比多布拉",
    "SVC : 萨尔瓦多科朗，萨尔瓦多科朗",
    "SYP : 叙利亚镑，叙利亚镑",
    "SZL : 斯威士兰里兰吉尼，埃斯瓦蒂尼里兰吉尼",
    "THB : 泰铢，泰国铢",
    "TJS : 塔吉克斯坦索莫尼，塔吉克斯坦索莫尼",
    "TMT : 土库曼斯坦马纳特，土库曼斯坦马纳特",
    "TND : 突尼斯第纳尔，突尼斯第纳尔",
    "TOP : 汤加潘加，汤加潘加",
    "TRY : 土耳其里拉，土耳其里拉，北塞浦路斯",
    "TTD : 特立尼达和多巴哥元，特立尼达和多巴哥元，特立尼达，多巴哥",
    "TWD : 新台币，新台币",
    "TZS : 坦桑尼亚先令，坦桑尼亚先令",
    "UAH : 乌克兰格里夫纳，乌克兰格里夫纳",
    "UGX : 乌干达先令，乌干达先令",
    "USD : 美元，美元，美国，美属萨摩亚，美属维尔京群岛，英属印度洋领地，英属维尔京群岛，厄瓜多尔，萨尔瓦多，关岛，海地，密克罗尼西亚，北马里亚纳群岛，帕劳，巴拿马，波多黎各，特克斯和凯科斯群岛，美国本土外小岛屿，威克岛，东帝汶",
    "UYU : 乌拉圭比索，乌拉圭比索",
    "UZS : 乌兹别克斯坦苏姆，乌兹别克斯坦苏姆",
    "VEF : 委内瑞拉玻利瓦尔，委内瑞拉玻利瓦尔",
    "VND : 越南盾，越南盾",
    "VUV : 瓦努阿图瓦图，瓦努阿图瓦图",
    "WST : 萨摩亚塔拉，萨摩亚塔拉",
    "XAF : 中非法郎 BEAC，中非经济和货币共同体（BEAC）法郎 BEAC，喀麦隆，中非共和国，乍得，刚果/布拉柴维尔，赤道几内亚，加蓬",
    "XAG : 银盎司，银",
    "XAU : 金盎司，金",
    "XCD：东加勒比元，安圭拉，安提瓜和巴布达，多米尼克，格林纳达，圣文森特和格林纳丁斯，蒙特塞拉特",
    "XDR：IMF特别提款权，国际货币基金组织（IMF）特别提款权",
    "XOF：CFA法郎，西非金融共同体（BCEAO）法郎，贝宁，布基纳法索，科特迪瓦，几内亚比绍，马里，尼日尔，塞内加尔，多哥",
    "XPF：CFP法郎，太平洋法郎，法属波利尼西亚，新喀里多尼亚，瓦利斯和富图纳群岛",
    "YER：也门里亚尔，也门里亚尔",
    "ZAR：南非兰特，南非，莱索托，纳米比亚",
    "ZMK：赞比亚克瓦查，赞比亚克瓦查",
    "ZMW：赞比亚克瓦查，赞比亚克瓦查",
    "ZWL：津巴布韦元，津巴布韦元",
"""
# SEEM3650 Project

This project is predicting the number of nurses working in Hong Kong public hospitals who will leave the industry in the specific year.

## Usage

You can use the following command to predict the number leaving nurses in the sepcified year:

```bash
# Getting prediction result in 2024 and you can also predict other year
python main.py 2024

# The prediction result will be in 2024 by default if no year is specified
python main.py
```

## Project structure

| Filename                   | Description                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| [dataset.csv](dataset.csv) | Dataset file that contains data range from year 2009 to 2022                                    |
| [main.py](main.py)         | Python script that can be executed by command line to obtain prediction result on specific year |
| [main.ipynb](main.ipynb)   | Python notebook that shows the code from [main.py](main.py) in blocks structure                 |

## Data Source

- https://www.legco.gov.hk/yr10-11/chinese/panels/hs/papers/hs0711cb2-2298-3-c.pdf
- https://www.hk01.com/社會新聞/306910/醫護逃亡-護士-專職醫療流失率創8年新高-每16名護士1人離職
- https://www.am730.com.hk/column/新聞/引入護士刻不容緩/340010
- https://jump.mingpao.com/career-news/daily-news/【公立醫院醫護人手荒%E3%80%82護士流失率惡化】5醫院聯/
- https://www3.ha.org.hk/Data/HAStatistics/MajorReport?language=tc
- https://www.censtatd.gov.hk/en/data/stat_report/product/B1120017/att/B1120017052022XXXXB0100.pdf

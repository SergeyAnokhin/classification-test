from elasticsearch import Elasticsearch
from domain import Measure

class ElasticSearchHelper:
    def GetRange(self, deltaData):
        result = []
        for row in deltaData:
            result.append({"from": row.start.date, "to": row.end.date})
        return result

    
    def Search(self, request):
        es = Elasticsearch([{'host': 'search-es-domain-vu4rnb36e6pnimhdaaxry4jpyy.us-east-2.es.amazonaws.com', 'port': 80}])
        # es.get(index='history-2017.12.05', doc_type='measure', id='AWAmTXnszcI0DKot8mcZ')

        result = es.search(index="history-*", body=request)
        return result

    def ToMesure(self, esResult):
        data_pipetemp = {}
        for bucket in esResult['aggregations']['2']['buckets'][0]['3']['buckets']:
            days = (bucket['to'] - bucket['from']) / 1000 / 60 / 60 / 24
            data_pipetemp[bucket['key']] = { 'value': bucket['1']['value'], 'duration': days }
        return data_pipetemp
class OpenSearchBody:
    def __init__(self, start:float, end:float):
        """
        import datetime
        import pytz
        beijing_tz = pytz.timezone('Asia/Shanghai')
        start = datetime.datetime(2024, 6, 21, 23, 59, 50).astimezone(beijing_tz).timestamp()
        end = datetime.datetime(2024, 6, 22, 23, 59, 59).astimezone(beijing_tz).timestamp()
        """
        self.start = start
        self.end = end
        self.body = self.to_dict()
    
    def to_dict(self):
        return {
            "size": 100,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match_phrase": {
                                "rag_triggered": True
                            }
                        },
                        {
                            "range": {
                                "timestamp": {
                                    "gte": self.start,
                                    "lte": self.end
                                } 
                            }
                        }
                    ],
                }
            }
        }
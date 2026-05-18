import re
import pymysql.cursors
from pymongo.mongo_client import MongoClient
import ciso8601
import datetime
import ftfy
import textacy.preprocessing
import os
from pandas import read_csv

class JournalFilter:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'journal_cnt.csv')) as csvfile:
            self.journals = read_csv(csvfile)
    
    def voteFilter(self, pub, numVote):
        for line in self.journals.itertuples():
            if line.Email >= numVote and line.Publisher == pub:
                yield line.Journal


class DataBaseFetcher:

    def __init__(self):
        self.client = MongoClient('mongodb://%s:%s@infochain.ece.udel.edu:27017/' % ('crawl', 'infochain'))
        self.crossref_col = self.client['crawl']['Article_crossref']

        # self.connection = pymysql.connect(host='localhost',
        #                      user='root',
        #                      password='infodeep',
        #                      database='crawl',
        #                      charset='utf8mb4')

    # def parsed_doi(self, publisher):
    #     with self.connection.cursor() as cursor:
    #         sql = "SELECT DOI FROM `Record` where `Publisher` = %s And `Status` = 'Parse'"
    #         cursor.execute(sql, (publisher,))
    #         data = cursor.fetchall()
    #         doi = [i[0] for i in data]
    #         return doi

    # def locatePublisher(self, doi):
    #     with self.connection.cursor() as cursor:
    #         sql = "SELECT Publisher FROM `Record` where `DOI` = %s"
    #         cursor.execute(sql, (doi,))
    #         data = cursor.fetchone()
    #         publisher = [i[0] for i in data]
    #         return publisher[0]

    def retriveDataMongo(self, doi, publisher=None):
        if publisher is None:
            publisher = self.locatePublisher(doi)
        # if publisher == 'RSC':
        #     doi = 'https://doi.org/' + doi.upper()
        mongo_col = self.client['crawl'][f'{publisher}_parse']
        item = mongo_col.find_one({'DOI': doi})
        if item is None:
            return None
        assert publisher == item["Publisher"]
        if item["Title"] is None:
            return None
        title = item["Title"]
        journal = item["Journal"][0] if isinstance(item["Journal"], list) else item["Journal"]
        section = item["Seccion"]
        data = {"publisher": publisher, "title": title, "journal": journal, "section": section}
        return data
    
    def iterJournalData(self, publisher, journal=None):
        mongo_col = self.client['crawl'][f'{publisher}_parse']
        if journal is None:
            iterator = mongo_col.find()
        else:
            iterator = mongo_col.find({'Journal': journal})
        for item in iterator:
            doi = item['DOI']
            if publisher == 'RSC':
                doi = doi.replace('https://doi.org/', '').lower()
            elif publisher == 'ACS':
                try:
                    doi = doi[0]
                except IndexError:
                    continue
            assert publisher == item["Publisher"]
            if item["Title"] is None:
                continue
            title = item["Title"]
            journal = item["Journal"][0] if isinstance(item["Journal"], list) else item["Journal"]
            section = item["Seccion"]
            yield {"doi": doi, "publisher": publisher, "title": title, "journal": journal, "section": section}

    def crossrefData(self, doi):
        #DOI names are case insensitive, using ASCII case folding for comparison of text
        itemCross = self.crossref_col.find_one({'DOI': doi.lower()}, {
            "created": True,
            "author": True,
            "is-referenced-by-count": True
        })
        # print(item["created"])
        # {'date-parts': [[2019, 6, 13]], 'date-time': '2019-06-13T18:34:53Z', 'timestamp': 1560450893000}
        # dt = ciso8601.parse_datetime(item["created"]["date-time"])
        # print(int(dt.timestamp() * 1000))
        # 1560450893000
        # this is why we use 1000
        if itemCross != None:
            if "timestamp" in itemCross["created"]:
                timestamp = itemCross["created"]["timestamp"]
            elif "date-time" in itemCross["created"]:
                dt = ciso8601.parse_datetime(itemCross["created"]["date-time"])
                timestamp = int(dt.timestamp() * 1000)
            else:
                dt = datetime.datetime(year=2100, month=1, day=1)
                timestamp = int(dt.timestamp() * 1000)
            authors = [
                ' '.join([names.get("given", "Unknown"), names.get("family", "")])
                for names in itemCross.get("author", [])
            ]
            citation = itemCross.get("is-referenced-by-count", 0)

        else:
            #TODO always shown none timestamp result in frontend - check if it is 2100
            dt = datetime.datetime(year=2100, month=1, day=1)
            timestamp = int(dt.timestamp() * 1000)
            authors = []
            citation = 0
        return timestamp, authors, citation


class Normalizer:

    def __init__(self, fix_text=True, unicode=True, quotation=False, whitespace=True, email=False, url=False):
        pipeline = []
        if fix_text:
            pipeline.append(ftfy.fix_text)
        if unicode:
            pipeline.append(textacy.preprocessing.normalize.unicode)
        if quotation:
            pipeline.append(textacy.preprocessing.normalize.quotation_marks)
        if whitespace:
            pipeline.append(textacy.preprocessing.normalize.whitespace)
        if email:
            pipeline.append(textacy.preprocessing.replace.emails)
        if url:
            pipeline.append(textacy.preprocessing.replace.urls)
        self.pipeline = textacy.preprocessing.make_pipeline(*pipeline)

    # from https://github.com/CederGroupHub/TextCleanUp/blob/master/text_cleanup/text_cleanup.py
    # maybe too agressive
    @staticmethod
    def gen_hyphen_normalize_fn():
        hyphens = [173, 8722, ord('\ue5f8'), 727, 12287, 12257] + [i for i in range(8208, 8214)]
        re_str = ''.join([chr(c) for c in hyphens])
        re_str = '[' + re_str + ']'

        return lambda text: re.sub(re_str, chr(45), text)

    # from https://github.com/CederGroupHub/TextCleanUp/blob/master/text_cleanup/text_cleanup.py
    # maybe too agressive
    @staticmethod
    def gen_quote_normalize_fn():
        quotes_double = [171, 187, 8220, 8221, 8222, 8223, 8243]
        quotes_single = [8216, 8217, 8218, 8219, 8242, 8249, 8250]

        single_re_str = ''.join([chr(c) for c in quotes_single])
        single_re_str = '[' + single_re_str + ']'

        double_re_str = ''.join([chr(c) for c in quotes_double])
        double_re_str = '[' + double_re_str + ']'

        return lambda text: re.sub(double_re_str, chr(34), re.sub(single_re_str, chr(39), text))

    def __call__(self, text):
        return self.pipeline(text)


class ParagraphExtractor():

    def __init__(self, normalize=True):
        self.skip = set([
            "Subjects", "Conflicts of interest", "Graphical abstract", "Rights and permissions", "Data availability",
            "Change history", "Code availability"
        ])
        self.abstract_skip = set(['graphic', 'graphical abstract', 'graphic abstract'])
        self.normalizer = Normalizer() if normalize else None

    def recursive_extract(self, section, path, res):
        assert type(section) == dict
        path = (*path, section['name']) if section['name'] not in path else path  #check if the title is inside Title
        strbuf = []
        
        if 'content' in section:
            content = section['content']
            if type(content) == str:
                content = [content]
            elif type(content) == list:
                pass
            else:
                raise RuntimeError('unknown para type')
        else:
            content = []
        
        for subsection in content:
            if type(subsection) == str:
                strbuf.append(subsection)
            elif type(subsection) == dict:
                if strbuf:
                    res.append((path, tuple(strbuf)))
                    strbuf = []
                self.recursive_extract(subsection, path, res)
            else:
                raise RuntimeError('unknown para type')
        
        if strbuf:
            res.append((path, tuple(strbuf)))
            strbuf = []

    def raw_section(self, data):
        sections = []
        strbuf = []
        # some are empty DOI: "10.1016/j.cocis.2009.07.001" in Elsevier
        if data is None or 'section' not in data or data['section'] is None:
            return []

        for para in data['section']:
            if type(para) == dict:
                self.recursive_extract(para, tuple(), sections)
                if strbuf:
                    sections.append((('',), tuple(strbuf)))
                    strbuf = []
            elif type(para) == str:
                # Some articles have sections without title, for that reason we are save of
                strbuf.append(para)
            else:
                raise RuntimeError('unknown para type')
        if strbuf:
            sections.append((('',), tuple(strbuf)))
            strbuf = []

        for path, contents in sections:
            if len(contents) == 0 or sum(map(len, contents)) == 0:
                continue
            else:
                if self.normalizer:
                    contents = [self.normalizer(text) for text in contents]
                yield path, contents

    def __call__(self, data):
        abstract_sections, other_sections = [], []
        for path, contents in self.raw_section(data):
            if self.isAbstract(path):
                all_text = '\n'.join(contents)
                # if len(path) > 1:
                #     # add subtitle, but create a lot of short sentence so comment out
                #     all_text = ' '.join(path[1:]) + '\n' + all_text
                abstract_sections.append(all_text)
            elif path[0] not in self.skip:
                other_sections.append((path, contents))
        # ensure all abstract text are exported in one paragraph
        # TODO discuss whether we need to use first introduction as abstract
        if abstract_sections:
            yield (True, ('Abstract',), abstract_sections)
        for path, contents in other_sections:
            yield (False, path, contents)

    def isAbstract(self, path):
        # most use Abstract as top section title
        if 'Abstract' == path[0]:
            if len(path) > 1 and path[1].lower() in self.abstract_skip:
                return False
            else:
                return True
        else:
            return False
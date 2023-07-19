## nlp_parser Tutorial 
이거 왜 하냐면요 
크로링으로 긁어온 HTML 파일에서 abstract만 긁어서 json파일로 저장해주는겁니다


논문별로 가능 



```
import time
import os
from copy import copy

import re
import json

import requests
from bs4 import BeautifulSoup, element
from urllib.request import urlopen, urlretrieve
from PIL import Image
from io import BytesIO

import os
from glob import glob
import shutil
import re
import json
import numpy as np

import requests
from bs4 import BeautifulSoup

from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation

RE_ENTER = re.compile('\n')
RE_SPACES = re.compile('  +')


def get_soup_from_url(url, parser='lxml'):
    response = requests.get(url)
    return BeautifulSoup(response.content, parser)


def get_soup_from_html(filename, parser='lxml'):
    with open(filename, encoding='UTF-8') as f:
        html = f.read()
    html = re.sub(RE_ENTER, ' ', html)
    html = re.sub(RE_SPACES, ' ', html)

    soup = BeautifulSoup(html, parser)
    return soup


def normalize(string):
    string = re.sub(f'\d+|[{re.escape(punctuation)}]', '', string)
    tokens = word_tokenize(string.lower())
    string = ' '.join([WordNetLemmatizer().lemmatize(word) for word in tokens])

    return string


class Parser:
    def __init__(self):
        self.datatype = ['metadata', 'figures', 'tables', 'document']

        self.metadata = ['title', 'author', 'journal', 'doi', 'references']
        self.figure = ['title', 'caption', 'url']
        self.table = ['title', 'caption', 'url', 'tag']

        self.parser = 'lxml'
        self.img_downloader = urlretrieve

        self.htag = '<h2>{}</h2>'
        self.divtag = '<div>{}</div>'
        self.htmlschema = '<!DOCTYPE html><body>{}</body></html>'

        self.noise = []
        self._noise = None

        self.filetype = '\.\w+'
        self._filetype = re.compile(self.filetype)

    def update_options(self):
        self._noise[-1] = re.compile('|'.join(self.noise))
        self._filetype = re.compile(self.filetype)

    def _refine_text(self, string):
        return string

    def _get_text(self, x):
        if not x: return ''

        if isinstance(x, str):
            return self._refine_text(x)
        if isinstance(x, element.Tag):
            return self._refine_text(x.text)

        if isinstance(x, list):
            if len(x) == 1: return self._get_text(*x)

            if isinstance(x[0], str):
                return [self._refine_text(t) for t in x]
            if isinstance(x[0], element.Tag):
                return [self._refine_text(t.text) for t in x]

        if isinstance(x, dict):
            return {self._get_text(k): self._get_text(v) for k, v in x.items()}

    def _get_metadata(self, soup, data):
        # method overriding
        return data

    def get_metadata(self, soup):
        data = {k: None for k in self.metadata}
        data = self._get_text(self._get_metadata(soup, data))

        return data

    def _get_figures(self, soup, figure):
        # method overriding
        figures = {}
        figures[''] = figure
        return figures

    def get_figures(self, soup, savepath=None, display=True):
        fig = {k: None for k in self.figure}
        figures = self._get_text(self._get_figures(soup, fig))

        if savepath:
            for idx, data in figures.items():
                url = data['url']
                filetype = re.findall(self._filetype, url)[-1]

                try:
                    self.img_downloader(url, savepath + '/' + idx + filetype)
                except:
                    print(f'failed download: {url}')

        if display:
            for data in figures.values():
                url = data['url']
                print('\n'.join(list(data.values())))

                try:
                    with urlopen(url) as f:
                        with Image.open(BytesIO(f.read())) as img:
                            img.show()
                except:
                    print(f'failed loading: {url}')

        return figures

    def _get_tables(self, soup, table):
        # method overriding
        tables = {}
        tables[''] = table
        return tables

    def _get_table_htmlstr(self, tables):
        tags = ''
        for k, v in tables.items():
            tag = self.htag.format(k) + v['tag']
            tags += self.divtag.format(tag)

        htmlstr = self.htmlschema.format(tags)
        return htmlstr

    def get_tables(self, soup, savepath=None):
        tbl = {k: None for k in self.table}
        tables = self._get_text(self._get_tables(soup, tbl))

        if savepath:
            for idx, data in tables.items():
                for i, url in enumerate(data['url']):
                    filetype = re.findall(self._filetype, url)[-1]
                    try:
                        self.img_downloader(url, savepath + '/' + idx + f'_image{i + 1:02d}' + filetype)
                    except:
                        print(f'failed download: {url}')

            with open(savepath + '/table.html', 'w') as f:
                f.write(self._get_table_htmlstr(tables))

        return tables

    def _get_hierarchy(self, tags):
        '''
        h1 ~ h6 태그와 p 태그로 이루어진 tags list를 hierarchy를 가진 dict로 바꿔주는 함수.
        heading 수준에 따라 깊이가 정렬되고 맨 끝단 p 태그만 남았을 때 list를 반환한다.
        '''
        try:
            scale = {int(tag.name[1]) for tag in tags if tag.name != 'p'}  # 존재하는 h 태그 종류
            scale = min(scale)  # 최솟값 -> 가장 상위 태그(1부터 6까지 내려감)

        except ValueError:
            return self._get_text(tags)  # 최솟값 계산 실패 = heading 태그가 없음 -> 맨 끝

        key_idx = [i for i, tag in enumerate(tags) if tag.name == 'h' + str(scale)]
        key_idx += [len(tags)]

        if tags[0].name == 'p':  # 맨 처음이 heading이 아닌 경우
            tmp_tag = copy(tags[key_idx[0]])
            tmp_tag.string = ''

            tags.insert(0, tmp_tag)
            key_idx = [0] + list(map(lambda x: x + 1, key_idx))

        out = {}
        for i in range(1, len(key_idx)):
            start, end = key_idx[i - 1:i + 1]

            k = self._get_text(tags[start]).strip()
            v_tags = tags[start + 1:end]

            out[k] = self._get_hierarchy(v_tags)

        return out

    def _get_document(self, soup):
        # method overriding
        document = {}
        document[''] = ''
        return document

    def get_document(self, soup):
        document = self._get_text(self._get_document(soup))
        return document

    def parse(self, soup, target=None, savepath=None, display=False):
        if not target: target = self.datatype

        result = {}
        if 'metadata' in target:
            result['metadata'] = self.get_metadata(soup)
        if 'figures' in target:
            result['figures'] = self.get_figures(soup, savepath, display)
        if 'tables' in target:
            result['tables'] = self.get_tables(soup, savepath)
        if 'document' in target:
            result['document'] = self.get_document(soup)

        return result

    def run(self, path, srcdir, target=None, test=True, download=True):
        starttime = time.time()
        srcfiles = [filename for filename in os.listdir(path + '/' + srcdir)
                    if filename[-5:] == '.html']

        savepath = path
        if test:
            savepath += '/' + srcdir + time.strftime('_%y%m%d%H%M%S', time.localtime(starttime))
            os.mkdir(savepath)

        if not target: target = self.datatype

        existing = {dirname: [] for dirname in ['data', 'figure', 'table']}
        try:
            os.mkdir(savepath + '/' + 'data')
            if download:
                if 'figures' in target: os.mkdir(savepath + '/' + 'figure')
                if 'tables' in target: os.mkdir(savepath + '/' + 'table')
        except:
            existing = {dirname: os.listdir(savepath + '/' + dirname) for dirname in existing.keys()}

        for srchtml in srcfiles:
            print(srchtml)
            savename = srchtml[:-5]
            soup = get_soup_from_html('/'.join([path, srcdir, srchtml]), self.parser)
            print("soup:\n", soup)
            jsonfile = savename + '.json'
            jsonpath = '/'.join([savepath, 'data', jsonfile])
            try:
                article_data = self.parse(soup, target)

            except:
                print("article error!!!")
                continue
            if jsonfile in existing['data']:  # 기존 파일 존재 시
                with open(jsonpath, 'r', encoding='UTF8') as f:
                    existing_json = json.load(f)

                count = [t in existing_json.keys() for t in target]  # target type이 다 있는지 확인

                if len(count) - sum(count) != 0:  # 다 있지 않을 때만 저장
                    with open(jsonpath, 'w', encoding='UTF8') as f:
                        json.dump(article_data, f, indent=4, ensure_ascii=False)

            else:  # 기존 파일 없음 -> 저장
                with open(jsonpath, 'w', encoding='UTF8') as f:
                    json.dump(article_data, f, indent=4, ensure_ascii=False)

            if not download: continue

            if 'figures' in target and article_data['figures']:

                for idx, data in article_data['figures'].items():

                    url = data['url']

                    if isinstance(url, str):

                        filetype = re.findall(self._filetype, url)[-1]

                        imgfile = savename + '_' + idx + filetype
                        imgpath = '/'.join([savepath, 'figure', imgfile])

                        if imgfile not in existing['figure']:  # 이미지 없을 때만 저장
                            try:
                                self.img_downloader(url, imgpath)
                            except:
                                print(f'failed download: {url}')

                    elif isinstance(url, list):

                        for url_ in url:

                            filetype = re.findall(self._filetype, url_)[-1]

                            imgfile = savename + '_' + idx + filetype
                            imgpath = '/'.join([savepath, 'figure', imgfile])

                            if imgfile not in existing['figure']:  # 이미지 없을 때만 저장
                                try:
                                    self.img_downloader(url_, imgpath)
                                except:
                                    print(f'failed download: {url}')

            if 'tables' in target and article_data['tables']:

                htmlstr = self._get_table_htmlstr(article_data['tables'])
                htmlfile = savename + '.html'
                htmlpath = '/'.join([savepath, 'table', htmlfile])

                is_existing = False

                if htmlfile in existing['table']:
                    with open(htmlpath, 'r', encoding='UTF8') as f:
                        is_existing = htmlstr == f.read()

                if not is_existing:
                    htmlstr = BeautifulSoup(htmlstr, 'lxml').prettify()
                    with open(htmlpath, 'w', encoding='UTF8') as f:
                        f.write(htmlstr)

                for idx, data in article_data['tables'].items():
                    if not data['url']: continue

                    urls = data['url'] if isinstance(data['url'], list) else [data['url']]

                    for i, url in enumerate(urls):

                        filetype = re.findall(self._filetype, url)[-1]

                        imgfile = savename + '_' + idx + f'_image{i + 1:02d}' + filetype
                        imgpath = '/'.join([savepath, 'table', imgfile])

                        if imgfile not in existing['table']:
                            try:
                                self.img_downloader(url, imgpath)
                            except:
                                print(f'failed download: {url}')


class NatureParser(Parser):
    def __init__(self):
        super().__init__()

        self.noise_sections = [
            'data availability',
            'code availability',
            'change history',
            'references',
            'acknowledgements',
            'author information',
            'electronic supplementary material',
            'ethics declarations',
            'additional information',
            'supplementary information',
            'source data',
            'rights and permissions',
            'about this article',
            'comments'
        ]

    def _refine_text(self, string):
        string = super()._refine_text(string)

        string = re.sub(',+\.', '.', string)
        string = re.sub(',+', ',', string)

        return string

    def _get_metadata(self, soup, data):
        data['title'] = soup.select_one('.c-article-title')
        data['author'] = soup.select('.c-article-author-list__item > a')
        data['journal'] = soup.select_one('.c-article-info-details i')

        data['doi'] = soup.select_one('.c-bibliographic-information__value a')
        data['references'] = soup.select('.c-article-references__text')

        data['citation'] = soup.select_one('.c-bibliographic-information__citation')

        return data

    def _get_figures(self, soup, figure):
        figures = {}

        for tag in soup.select('.c-article-section__figure'):
            idx = tag.get('id')

            fig = figure.copy()

            try:
                fig['title'] = tag.select_one('figcaption')
                fig['caption'] = tag.select_one('#' + idx + '-desc')

                url = 'https://www.nature.com' + tag.select_one('a').attrs['href']
                fig_soup = get_soup_from_url(url)

                fig['url'] = 'https:' + fig_soup.select_one('figure img').attrs['src']

            except (TypeError, AttributeError):
                idx = 'graphical abstract'
                fig['url'] = 'https:' + tag.select_one('figure img').attrs['src']

            figures[idx] = fig

        return figures

    def _get_tables(self, soup, table):
        tables = {}

        for tag in soup.select('.c-article-table'):
            idx = tag.get('id')

            url = 'https://www.nature.com' + tag.select_one('a').attrs['href']
            tbl_soup = get_soup_from_url(url)

            tbl = table.copy()

            tbl['title'] = tbl_soup.select_one('.c-article-table-title')
            tbl['caption'] = tbl_soup.select('.c-article-table-footer li')

            tbl_tag = tbl_soup.select_one('.c-article-table-container table')
            tbl['url'] = ['https:' + img.attrs['src'] for img in tbl_tag.select('img')]
            tbl['tag'] = str(tbl_tag)

            tables[idx] = tbl

        return tables

    def _get_document(self, soup):
        soup = copy(soup)

        for tag in soup.select('.c-article-section__figure, .c-article-table'):
            tag.decompose()

        document = {}

        for sect in soup.select('#content section'):
            title = sect.get('data-title')
            if title and title.lower() in self.noise_sections: continue

            title = sect.select_one('.c-article-section__title')
            if not title or title.text == 'Further reading': continue

            for a in sect.select('a'):
                aria_label = a.get('data-test')
                if aria_label and aria_label == 'citation-ref':
                    a.decompose()

            content = sect.select_one('.c-article-section__content').select('h3, h4, h5, h6, p')
            document[title] = self._get_hierarchy(content)

        return document


class RSCParser(Parser):
    def __init__(self):
        super().__init__()

        self.noise_headings = [
            'conflict of interest',
            'author contribution',
            'acknowledgement',
            'note and reference',
            'reference',
            'footnote'
        ]

    def _refine_reftext(self, tag, join='/'):
        tokens = tag.get_text(join).split(join)
        tokens = [re.sub('  +', ' ', token).lstrip() for token in tokens]

        string = ''.join(tokens)
        string = string.replace(' —', '—').replace(' . ', '.')

        return string

    def _get_metadata(self, soup, data):
        data['title'] = self._get_text(soup.select_one('#maincontent .article__title')).strip()

        authors = soup.select('.article__author-link')
        data['author'] = [author.select_one('a') for author in authors]

        data['journal'] = soup.select_one('.article-nav h3')
        data['doi'] = self._get_text(soup.select_one('.doi-link')).strip()

        references = []
        for refer in soup.select('.ref-list li'):
            for _ in refer.select('a'):
                refer.a.decompose()

            references.append(self._refine_reftext(refer))

        data['references'] = references

        return data

    def _get_figures(self, soup, figure):
        figures = {}

        # graphical abstract: download X
        toc = soup.select_one('.capsule__article-image img') if False else None
        if toc:  # url이 이미지 확장자로 되어있지 않음 -> 다운로드는 안되는데 그림판 붙여넣기는 됨. byteIO?
            idx = toc.get('title')
            fig = figure.copy()

            fig['url'] = 'https://pubs.rsc.org' + toc.attrs['src']

            figures[idx] = fig

        for tag in soup.select('.img-tbl'):
            idx = tag.get('id')
            fig = figure.copy()

            fig['title'] = self._get_text(tag.select_one('.section-number')).strip()
            if not fig['title']: continue  # 수식이 이미지로 삽입된 경우 있음

            fig['caption'] = self._get_text(tag.select('figcaption > span')[1:]).strip()
            fig['url'] = 'https://pubs.rsc.org' + tag.select_one('img').attrs['data-original']

            figures[idx] = fig

        return figures

    def _get_tables(self, soup, table):
        tables = {}

        for tag in soup.select('.pnl--table'):
            idx = tag.get('id')

            tbl = table.copy()

            tbl['title'] = copy(tag)  # 내부에 plain text랑 같이 table 태그 있음
            tbl['title'].span.string += ' '
            if tbl['title'].div: tbl['title'].div.decompose()

            caption = tag.select_one('tfoot')
            if caption:
                last_key = ''
                tbl['caption'] = {last_key: []}

                for child in caption.select('.tfootnote, .sup_inf'):
                    if 'tfootnote' in child.get('class'):
                        last_key = child
                        tbl['caption'][last_key] = []
                    else:
                        tbl['caption'][last_key].append(child)

                if tbl['caption'][''] == []:
                    del tbl['caption']['']

            tbl['tag'] = str(tag.select_one('table'))

            tables[idx] = tbl

        return tables

    def _get_document(self, soup):
        soup = copy(soup)

        abstract = [soup.select_one('.article-abstract__heading')]
        abstract += soup.select('.capsule__column-wrapper p')
        abstract = self._get_hierarchy(abstract)

        context = soup.select_one('#pnlArticleContent .pnl--box')
        if context:
            context.decompose()

        document = []
        append_p = True
        for tag in soup.select_one('#pnlArticleContent').select('h2, h3, h4, h5, h6, p'):
            if tag.name != 'p':
                if normalize(tag.text) in self.noise_headings:
                    append_p = False
                    continue
                else:
                    append_p = True
                    section_num = tag.select_one('.section-number')

                    if section_num:
                        section_num.string += ' '

                    document.append(tag)

            elif append_p:
                for a in tag.select('a'):
                    if '#cit' in a.get('href'):
                        a.decompose()

                if re.match('This journal is © The Royal Society of Chemistry', tag.text): continue
                document.append(tag)

        document = self._get_hierarchy(document) if document else {}
        if isinstance(document, list): document = {'': document}
        document = dict(abstract, **document)

        return document


class ElsevierParser(Parser):
    def __init__(self):
        super().__init__()

        self.parser = 'html.parser'

        def img_downloader(url, savename):
            with Image.open(BytesIO(requests.get(url).content)) as im:
                im.save(savename)

        self.img_downloader = img_downloader

        self.noise_headings = [
            'credit authorship contribution statement',
            'credit author statement',
            'author contribution',
            'conflict of interest',
            'declaration of competing interest',
            'author statement'
        ]

    def _refine_text(self, string):
        string = super()._refine_text(string)

        string = re.sub(' \[[, ]*?\]', '', string)
        string = re.sub(' , (, )*', '', string)
        string = re.sub(' ,', ',', string)
        string = re.sub(', ?\.', '.', string)
        string = re.sub(' \.', '.', string)

        return string

    def _get_metadata(self, soup, data):
        data['title'] = soup.select_one('.title-text')

        author = []
        for ath in soup.select('#author-group .content'):
            for span in ath.select('.author-ref'):
                span.decompose()
            author.append(ath.get_text(' '))

        data['author'] = author

        data['journal'] = soup.select_one('#publication-title')
        data['doi'] = soup.select_one('#doi-link .doi')

        references = []
        for refer in soup.select('.reference'):
            ref_links = refer.select_one('.ReferenceLinks')
            if ref_links: ref_links.decompose()
            references.append(refer.get_text(' '))

        data['references'] = references

        data['keywords'] = soup.select('.keyword span')

        return data

    def _get_figures(self, soup, figure):
        figures = {}

        for tag in soup.select('figure'):
            idx = tag.get('id')

            fig = figure.copy()

            if tag.select_one('.captions .label'):
                tmp_tag = copy(tag)
                fig['title'] = tmp_tag.select_one('.captions .label').extract()
                fig['caption'] = tmp_tag.select_one('.captions').text[2:]

            else:  # graphical abstract: 보통 img만 있음
                fig['title'] = 'graphical abstract'
                fig['caption'] = tag.select_one('.captions')

            if 'inline-figure' in tag.get('class'):
                continue  # 본문 내 자료 그림도 넣어야하는지? 캡션 달린 item 아님.
                # fig['url'] = tag.select_one('img').attrs['src']
            else:
                fig['url'] = [link.attrs['href'] for link in tag.select('.download-link') if
                              'high-res' in link.get('title')]
                if not fig['url']:
                    fig['url'] = [link.attrs['href'] for link in tag.select('.download-link')]

            figures[idx] = fig

        return figures

    def _get_tables(self, soup, table):
        tables = {}

        for tag in soup.select('.tables'):
            idx = tag.get('id')

            tbl = table.copy()

            tbl['title'] = tag.select_one('.captions')
            tbl['caption'] = tag.select('.legend')

            tbl['tag'] = str(tag.select_one('table'))

            tables[idx] = tbl

        return tables

    def _get_document(self, soup):
        soup = copy(soup)
        for reference in soup.select('a.workspace-trigger'):
            if '#bib' in reference.get('href'):
                reference.decompose()
            elif '#b' in reference.get('href') and 'workspace-trigger' in reference.get('class'):
                reference.decompose()

        document = {}

        for abstract in soup.select('.Abstracts .abstract'):
            if 'author-highlights' in abstract.get('class'):
                content = abstract.select('h2, h3, h4, h5, h6, dd p')
            else:
                if 'graphical' in abstract.get('class') and abstract.select_one('ol'):
                    abstract.select_one('ol').decompose()
                content = abstract.select('h2, h3, h4, h5, h6, p')

            content = self._get_hierarchy(content)
            if not isinstance(content, dict): content = {'': content}

            document.update(content)

        content = copy(soup.select_one('#body div'))
        if content:
            for item in content.select('.tables, figure'):
                item.decompose()

            content = self._get_hierarchy(content.select('h2, h3, h4, h5, h6, p'))
            if not isinstance(content, dict): content = {'': content}  # 소제목이 없는 논문

        else:
            content = {}

        content = {k: v for k, v in content.items() if normalize(k) not in self.noise_headings}

        document = dict(document, **content)

        return document

    def run(self, path, srcdir, target=None, test=True, download=True):
        starttime = time.time()
        srcfiles = [filename for filename in os.listdir(path + '/' + srcdir)
                    if filename[-5:] == '.html']

        savepath = path
        if test:
            savepath += '/' + srcdir + time.strftime('_%y%m%d%H%M%S', time.localtime(starttime))
            os.mkdir(savepath)

        if not target: target = self.datatype

        existing = {dirname: [] for dirname in ['data', 'figure', 'table']}
        try:
            os.mkdir(savepath + '/' + 'data')
            if download:
                if 'figures' in target: os.mkdir(savepath + '/' + 'figure')
                if 'tables' in target: os.mkdir(savepath + '/' + 'table')
        except:
            existing = {dirname: os.listdir(savepath + '/' + dirname) for dirname in existing.keys()}

        for srchtml in srcfiles:
            print(srchtml)
            savename = srchtml[:-5]
            soup = get_soup_from_html('/'.join([path, srcdir, srchtml]), self.parser)

            jsonfile = savename + '.json'
            jsonpath = '/'.join([savepath, 'data', jsonfile])

            article_data = self.parse(soup, target)
            if jsonfile in existing['data']:  # 기존 파일 존재 시
                with open(jsonpath, 'r', encoding='UTF8') as f:
                    existing_json = json.load(f)

                count = [t in existing_json.keys() for t in target]  # target type이 다 있는지 확인

                if len(count) - sum(count) != 0:  # 다 있지 않을 때만 저장
                    with open(jsonpath, 'w', encoding='UTF8') as f:
                        json.dump(article_data, f, indent=4, ensure_ascii=False)

            else:  # 기존 파일 없음 -> 저장
                with open(jsonpath, 'w', encoding='UTF8') as f:
                    json.dump(article_data, f, indent=4, ensure_ascii=False)

            if not download: continue

            if 'figures' in target and article_data['figures']:

                for idx, data in article_data['figures'].items():
                    url = data['url']
                    if isinstance(url, str):
                        filetype = re.findall(self._filetype, url)[-1]

                        imgfile = savename + '_' + idx + filetype
                        imgpath = '/'.join([savepath, 'figure', imgfile])

                        if imgfile not in existing['figure']:  # 이미지 없을 때만 저장
                            try:
                                self.img_downloader(url, imgpath)
                            except:
                                print(f'failed download: {url}')

                    elif isinstance(url, list):
                        for url_ in url:
                            filetype = re.findall(self._filetype, url_)[-1]

                            imgfile = savename + '_' + idx + filetype
                            imgpath = '/'.join([savepath, 'figure', imgfile])

                            if imgfile not in existing['figure']:  # 이미지 없을 때만 저장
                                try:
                                    self.img_downloader(url_, imgpath)
                                except:
                                    print(f'failed download: {url}')

            if 'tables' in target and article_data['tables']:
                htmlstr = self._get_table_htmlstr(article_data['tables'])
                htmlfile = savename + '.html'
                htmlpath = '/'.join([savepath, 'table', htmlfile])

                is_existing = False

                if htmlfile in existing['table']:
                    with open(htmlpath, 'r', encoding='UTF8') as f:
                        is_existing = htmlstr == f.read()

                if not is_existing:
                    htmlstr = BeautifulSoup(htmlstr, 'lxml').prettify()
                    with open(htmlpath, 'w', encoding='UTF8') as f:
                        f.write(htmlstr)

                for idx, data in article_data['tables'].items():
                    if not data['url']: continue

                    urls = data['url'] if isinstance(data['url'], list) else [data['url']]

                    for i, url in enumerate(urls):

                        filetype = re.findall(self._filetype, url)[-1]

                        imgfile = savename + '_' + idx + f'_image{i + 1:02d}' + filetype
                        imgpath = '/'.join([savepath, 'table', imgfile])

                        if imgfile not in existing['table']:
                            try:
                                self.img_downloader(url, imgpath)
                            except:
                                print(f'failed download: {url}')


class ScienceParser(Parser):
    def _refine_text(self, string):
        string = super()._refine_text(string)

        string = re.sub(' \((, )*\)', '', string)
        string = re.sub(' ?(–)', '', string)

        return string

    def _get_metadata(self, soup, data):
        data['title'] = soup.select_one('.article__headline')
        data['author'] = soup.select('.contributor-list li .name')

        metaline = soup.select_one('.meta-line').get_text('#').split('#')
        data['journal'] = metaline[1][:-1]
        data['doi'] = 'https://doi.org/' + metaline[-1][5:]

        references = []
        for refer in soup.select('.cit-metadata'):
            if 'unstructured' in refer.get('class'):
                references.append(self._get_text(refer))
                continue

            for ref_links in refer.select('.cit-pub-id, .cit-pub-id-sep'):
                ref_links.decompose()

            ref_authors = self._get_text(refer.select('.cit-auth-list li'))
            if isinstance(ref_authors, str): ref_authors = [ref_authors]
            ref_authors = ''.join([author[1:] for author in ref_authors])

            references.append(ref_authors + self._get_text(refer.select_one('cite')))

        data['references'] = references

        return data

    def _get_figures(self, soup, figure):
        figures = {}

        for tag in soup.select('figure'):
            idx = tag.get('id')

            fig = figure.copy()

            title = tag.select_one('figcaption').select('.fig-label, .caption-title')
            fig['title'] = ' '.join(self._get_text(title))

            fig['caption'] = tag.select('figcaption p')

            url = tag.select('.figure__options li')[1]
            if url: fig['url'] = url.select_one('a').attrs['href']

            figures[idx] = fig

        return figures

    def _get_tables(self, soup, table):
        tables = {}

        for tag in soup.select('.table'):
            # 저널마다 호스트 주소가 다름
            metaline = soup.select_one('.meta-line').get_text('#').split('#')
            journal = metaline[1][:-1]
            journal = 'science' if journal == 'Science' else 'advances'

            idx = tag.get('id')
            title = tag.select_one('.table-caption').select('.table-label, .caption-title')

            tbl = table.copy()

            tbl['title'] = ' '.join(self._get_text(title))
            tbl['caption'] = tag.select('.table-caption p') + tag.select('.table-foot p')
            tbl['tag'] = str(tag.select_one('table'))

            tables[idx] = tbl

        return tables

    def _get_document(self, soup):
        soup = copy(soup)

        for tag in soup.select('figure, .table, journal-interstitial, .app, .ref-list, .license, a.xref-bibr'):
            tag.decompose()

        document = {}
        for abstract in soup.select('.editor-summary, .abstract'):
            section = [
                abstract.select_one('h2, h3, h4, h5, h6'),
                *[p for p in abstract.select('p') if 'this issue' not in self._get_text(p)]
            ]

            document.update(self._get_hierarchy(section))

            abstract.decompose()

        content = soup.select_one('.fulltext-view').select('h2, h3, h4, h5, h6, p')
        content = self._get_hierarchy(content)
        if isinstance(content, list): content = {'': content}

        document.update(content)

        return document


class WileyParser(Parser):
    def __init__(self):
        super().__init__()

    def _refine_text(self, string):
        string = super()._refine_text(string)

        string = re.sub('\[\]', '', string)
        string = re.sub('(, )+ ', ' ', string)
        string = re.sub('\.-', '.', string)

        return string

    def _get_metadata(self, soup, data):
        data['title'] = soup.select_one('.citation__title')
        data['author'] = soup.select('#sb-1 .author-name')  # author 전부 안 나옴

        data['journal'] = soup.select_one('.journal-banner-image').get('alt')
        data['doi'] = soup.select_one('.epub-doi')

        references = []
        # 여러개 붙어있는 reference 한번에 처리됨, 반점 붙어있는 경우 있음.
        for refer in soup.select('#references-section li'):
            bullet = refer.select_one('.bullet')
            if bullet: bullet.decompose()

            for link in refer.select('.extra-links'):
                link.decompose()

            references.append(self._get_text(refer))

        data['references'] = references

        data['article type'] = soup.select_one('.doi-access-container .primary-heading')
        return data

    def _get_figures(self, soup, figure):
        soup = copy(soup)
        soup.select_one('#figure-viewer').decompose()

        figures = {}

        for tag in soup.select('figure'):
            idx = tag.get('id')

            fig = figure.copy()

            fig['title'] = tag.select_one('.figure__title')
            fig['caption'] = tag.select_one('figcaption .figure__caption')
            fig['url'] = 'https://chemistry-europe.onlinelibrary.wiley.com' + tag.select_one('a').attrs['href']

            figures[idx] = fig

        return figures

    # caption 추가 필요
    def _get_tables(self, soup, table):
        soup = copy(soup)

        tables = {}

        for tag in soup.select('.article-table-content'):
            idx = tag.get('id')

            tbl = table.copy()

            tbl['title'] = tag.select_one('header')

            tbl['tag'] = str(tag.select_one('table'))

            tables[idx] = tbl

        return tables

    def _get_document(self, soup):
        soup = copy(soup)

        for figure in soup.select('figure'):
            figure.decompose()

        for table in soup.select('.article-table-content'):
            table.decompose()

        for reference in soup.select('a.bibLink'):
            reference.decompose()

        abstract = []
        for abst in soup.select('.article-section__abstract'):

            if 'graphical-abstract' in abst.select_one('div').get('class') and soup.select_one(
                    '.journal-banner-image').get('alt') != 'Angewandte Chemie':
                abst.select_one('h2, h3, h4, h5, h6').string = 'Graphical Abstract'
                for p in abst.select('p'):
                    if p.text == '': p.decompose()

            abstract += abst.select('h2, h3, h4, h5, h6, p')

        abstract = self._get_hierarchy(abstract)

        content = {}
        for sect in soup.select('.article-section__full .article-section__content'):

            sect = self._get_hierarchy(sect.select('h2, h3, h4, h5, h6, p'))
            if not isinstance(sect, dict): sect = {'': sect}

            content.update(sect)

        return dict(abstract, **content)


class ACSParser(Parser):
    def __init__(self):
        super().__init__()

        def img_downloader(url, savename):
            with Image.open(BytesIO(requests.get(url).content)) as im:
                im.save(savename)

        self.img_downloader = img_downloader

    def _refine_text(self, string):
        string = super()._refine_text(string)

        string = re.sub(' \((, )*\)', '', string)
        string = re.sub(' (-)', '', string)

        return string

    def _get_metadata(self, soup, data):
        data['title'] = soup.select_one('.article_header-title')
        data['author'] = soup.select('.loa .hlFld-ContribAuthor')
        data['journal'] = soup.select_one('.aJhp_link')
        data['doi'] = soup.select_one('.article_header-doiurl')

        references = []
        for refer in soup.select('#references .NLM_citation'):
            refer.select_one('.citationLinks').decompose()

            references.append(refer)

        data['references'] = references

        return data

    def _get_figures(self, soup, figure):
        soup = copy(soup)
        soup.select_one('.tab__pane-figures').decompose()
        soup.select_one('.NLM_back').decompose()

        figures = {}

        for tag in soup.select('figure'):
            idx = tag.get('id')

            if 'article_abstract-img' in tag.get('class'):
                idx = 'graphical abstract'

            fig = figure.copy()

            fig['title'] = tag.select_one('figcaption')

            fig['caption'] = {}

            for child in tag.select('p.last'):
                key = child.select_one('.fn-label')
                if not key: continue

                key.extract()

                fig['caption'][key] = child

            fig['url'] = 'https://pubs.acs.org' + tag.select_one('.download-hi-res-img a').attrs['href']

            figures[idx] = fig

        return figures

    def _get_tables(self, soup, table):
        tables = {}

        for tag in soup.select('.NLM_table-wrap'):
            idx = tag.get('id')

            tbl = table.copy()

            tbl['title'] = tag.select_one('.NLM_caption')

            caption = tag.select_one('.NLM_table-wrap-foot')

            if caption:
                tbl['caption'] = {}

                for child in caption.select('.footnote'):
                    key = self._get_text(child.select_one('sup'))
                    val = self._get_text(child.select('p'))

                    tbl['caption'][key] = val

            tbl['tag'] = str(tag.select_one('table'))

            tables[idx] = tbl

        return tables

    def _get_document(self, soup):
        soup = copy(soup)

        soup.select_one('.NLM_back').decompose()
        soup.select_one('.articleCitedByDropzone').decompose()

        for tag in soup.select('figure, .NLM_table-wrap'):
            tag.decompose()

        for tag in soup.select('a.ref'):
            if 'internalNav' not in tag.get('class'):
                tag.decompose()

        abstract = []
        for sect in soup.select('.article_abstract, .synopsis'):
            abstract += sect.select('h2, h3, h4, h5, h6, p')

        abstract = self._get_hierarchy(abstract)

        content = {}
        for nlm_p in soup.select('.NLM_p'):
            nlm_p.name = 'p'

        content = soup.select('.NLM_sec h2, .NLM_sec h3, .NLM_sec h4, .NLM_sec h5, .NLM_sec h6, .NLM_p')
        if not content:
            content = soup.select_one('.article_content-left').select('h2, h3, h4, h5, h6, p')

        content = self._get_hierarchy(content)
        if isinstance(content, list): content = {'': content}

        return dict(abstract, **content)


if __name__ == '__main__':
    print(time.strftime('%c', time.localtime(time.time())))

    path = './ACS'
    srcdir = 'html'

    target = None
    parser = ACSParser()
    starttime = time.time()
    parser.run(path, srcdir, target=target, test=True, download=True)

    print(time.strftime('%c', time.localtime(time.time())))
```


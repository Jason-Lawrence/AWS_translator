import concurrent.futures
import multiprocessing
import logging
import os

import boto3
import pandas as pd


LOGGING_FORMAT = '%(asctime)s %(level)s %(message)s'
MANAGER = multiprocessing.Manager()


class Translator:
    
    def __init__(self, file_path, src_lang='', trgt_lang=''):
        self.file_path = file_path
        logging.basicConfig(format=LOGGING_FORMAT)
        self.logger = logging.getLogger('Translator')
        self.cache = MANAGER.dict()
        self.src_lang = src_lang
        self.trgt_lang = trgt_lang
        self.client = boto3.client('translate')
        
    @property
    def file_path(self):
        return self._file_path
    
    @file_path.setter
    def file_path(self, file_path):
        if os.path.exists(file_path):
            self._file_path = file_path
        
        else:
            raise FileNotFoundError(f'File Not Found Error')
        
    def open_excel(self, sheet_name=None):
        """
        Open up the Excel file.
        
        Args:
            sheet_name (list[str | int]): Specify which sheets to open, 
                Defaults to None and all sheets will open. 
        
        Return:
            dfs (dict{'sheet_name'}: pd.DataFrame): A dictionary of DataFrames.
        """
        dfs = pd.read_excel(self.file_path, sheet_name=sheet_name)
        return dfs
    
    def translate(self, item):
        """Send API request and add response to the shared cache."""
        try:
            resp = self.client.translate_text(
                Text=item,
                SourceLanguageCode=self.src_lang,
                TargetLanguageCode=self.trgt_lang
            )
            self.cache[item] = resp['TranslatedText']
        
        except Exception as e:
            self.logger.error(f'Failed with error: {e}')
            self.logger.warning(f'Failed to translate item: {item}')
            
    def translate_async(self, items):
        """
        Spin up Threads to submit API requests Asynchronously.
        Using threads since this is I/O bound.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=500) as threadpool:
            for item in items:
                threadpool.submit(self.translate, item)
                
    def check_cache(self, items):
        """
        Check the dictionary keys to see if any of them match up 
        with the list of items that was passed in.
        
        Args:
            items (list[str]): list of text to translate. 
                Checking if any are already translated.
        
        Return:
            not_found (list[str]): List of items that are 
                not a key in the dictionary.
        """
        return list(set(items) - set(self.cache.keys()))
    
    def translate_DataFrame(self, key, df):
        """
        Find the unique values in the DataFrame that needs to be 
        translated. Check the shared cache for any items that have
        already been translated. Translate the items that aren't already.

        Args:
            key (str): The name for the worksheet,
            df (pd.DataFrame): The DataFrame to translate.
        
        Return:
            key (str): The name of the worksheet.
            translated_df (pd.DataFrame): The DataFrame translated.
        """
        headers = [header for header in df.columns if not header.isascii()]
        mask = df.map(lambda x: True if isinstance(x, str) 
                        and not x.isascii() else False)
        
        unique_items = set(df[mask].stack())
        all_items = list(unique_items.union(set(headers)))
        items_to_translate = self.check_cache(all_items)
        self.translate_async(items_to_translate)
        
        lookup_df = df[mask].map(lambda x: self.cache.get(x),
                                 na_action='ignore')
        
        translated_df = df.mask(mask, lookup_df)
        
        translated_df.rename(
            columns={old: self.cache.get(old, old) for old in df.columns},
            inplace=True
        )
        
        return key, translated_df
    
    def translate_excel(self):
        """
        Asynchronously translate an Excel file.
        Load all worksheets into their own Pandas DataFrames
        Translate DataFrames in Parallel.
        Write translated DataFrames to their own worksheet in a new excel file.
        """
        dfs = self.open_excel()
        
        worksheet_names = [name for name in dfs.keys() if not name.isascii()]
        
        worksheet_names_to_translate = self.check_cache(worksheet_names)
        
        if len(worksheet_names_to_translate) > 0:
            self.logger.info(f'Translating {len(worksheet_names_to_translate)} items')
            self.translate_async(worksheet_names_to_translate)
        
        args = [(self.cache.get(key, key), value) for key, value in dfs.items()]
        
        self.logger.info('Creating child process...')
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            results = pool.starmap(self.translate_DataFrame, args)
            translated_dfs = {key: value for key, value in results}
        
        with pd.ExcelWriter(f'translated_{self.file_path}') as writer:
            for name, df in translated_dfs.items():
                df.to_excel(writer, sheet_name=name)
                
        self.logger.info('Excel file has been translated!')
        

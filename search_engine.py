
"""
Search Engine Implementation
Information Analyst Track - Assignment 3

Features:
- Inverted index with Porter stemming
- tf-idf ranking with importance weighting
- Efficient retrieval (<2s response time)
"""

import json
import os
import re
import math
import pickle
from collections import defaultdict
from pathlib import Path
from html.parser import HTMLParser
from typing import Dict, List, Tuple, Set
import time

# Porter Stemmer implementation
class PorterStemmer:
    """Porter Stemmer algorithm for word normalization"""
    
    def __init__(self):
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0
    
    def cons(self, i):
        if self.b[i] == 'a' or self.b[i] == 'e' or self.b[i] == 'i' or self.b[i] == 'o' or self.b[i] == 'u':
            return 0
        if self.b[i] == 'y':
            if i == self.k0:
                return 1
            else:
                return (not self.cons(i - 1))
        return 1

    def m(self):
        n = 0
        i = self.k0
        while 1:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i = i + 1
        i = i + 1
        while 1:
            while 1:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i = i + 1
            i = i + 1
            n = n + 1
            while 1:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i = i + 1
            i = i + 1

    def vowelinstem(self):
        for i in range(self.k0, self.j + 1):
            if not self.cons(i):
                return 1
        return 0

    def doublec(self, j):
        if j < (self.k0 + 1):
            return 0
        if (self.b[j] != self.b[j-1]):
            return 0
        return self.cons(j)

    def cvc(self, i):
        if i < (self.k0 + 2) or not self.cons(i) or self.cons(i-1) or not self.cons(i-2):
            return 0
        ch = self.b[i]
        if ch == 'w' or ch == 'x' or ch == 'y':
            return 0
        return 1

    def ends(self, s):
        length = len(s)
        if s[length - 1] != self.b[self.k]:
            return 0
        if length > (self.k - self.k0 + 1):
            return 0
        if self.b[self.k-length+1:self.k+1] != s:
            return 0
        self.j = self.k - length
        return 1

    def setto(self, s):
        length = len(s)
        self.b = self.b[:self.j+1] + s + self.b[self.j+length+1:]
        self.k = self.j + length

    def r(self, s):
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k = self.k - 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k = self.k - 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):   self.setto("ate")
            elif self.ends("bl"): self.setto("ble")
            elif self.ends("iz"): self.setto("ize")
            elif self.doublec(self.k):
                self.k = self.k - 1
                ch = self.b[self.k]
                if ch == 'l' or ch == 's' or ch == 'z':
                    self.k = self.k + 1
            elif (self.m() == 1 and self.cvc(self.k)):
                self.setto("e")

    def step1c(self):
        if (self.ends("y") and self.vowelinstem()):
            self.b = self.b[:self.k] + 'i' + self.b[self.k+1:]

    def step2(self):
        if self.b[self.k - 1] == 'a':
            if self.ends("ational"):   self.r("ate")
            elif self.ends("tional"):  self.r("tion")
        elif self.b[self.k - 1] == 'c':
            if self.ends("enci"):      self.r("ence")
            elif self.ends("anci"):    self.r("ance")
        elif self.b[self.k - 1] == 'e':
            if self.ends("izer"):      self.r("ize")
        elif self.b[self.k - 1] == 'l':
            if self.ends("bli"):       self.r("ble")
            elif self.ends("alli"):    self.r("al")
            elif self.ends("entli"):   self.r("ent")
            elif self.ends("eli"):     self.r("e")
            elif self.ends("ousli"):   self.r("ous")
        elif self.b[self.k - 1] == 'o':
            if self.ends("ization"):   self.r("ize")
            elif self.ends("ation"):   self.r("ate")
            elif self.ends("ator"):    self.r("ate")
        elif self.b[self.k - 1] == 's':
            if self.ends("alism"):     self.r("al")
            elif self.ends("iveness"): self.r("ive")
            elif self.ends("fulness"): self.r("ful")
            elif self.ends("ousness"): self.r("ous")
        elif self.b[self.k - 1] == 't':
            if self.ends("aliti"):     self.r("al")
            elif self.ends("iviti"):   self.r("ive")
            elif self.ends("biliti"):  self.r("ble")
        elif self.b[self.k - 1] == 'g':
            if self.ends("logi"):      self.r("log")

    def step3(self):
        if self.b[self.k] == 'e':
            if self.ends("icate"):     self.r("ic")
            elif self.ends("ative"):   self.r("")
            elif self.ends("alize"):   self.r("al")
        elif self.b[self.k] == 'i':
            if self.ends("iciti"):     self.r("ic")
        elif self.b[self.k] == 'l':
            if self.ends("ical"):      self.r("ic")
            elif self.ends("ful"):     self.r("")
        elif self.b[self.k] == 's':
            if self.ends("ness"):      self.r("")

    def step4(self):
        if self.b[self.k - 1] == 'a':
            if self.ends("al"): pass
            else: return
        elif self.b[self.k - 1] == 'c':
            if self.ends("ance"): pass
            elif self.ends("ence"): pass
            else: return
        elif self.b[self.k - 1] == 'e':
            if self.ends("er"): pass
            else: return
        elif self.b[self.k - 1] == 'i':
            if self.ends("ic"): pass
            else: return
        elif self.b[self.k - 1] == 'l':
            if self.ends("able"): pass
            elif self.ends("ible"): pass
            else: return
        elif self.b[self.k - 1] == 'n':
            if self.ends("ant"): pass
            elif self.ends("ement"): pass
            elif self.ends("ment"): pass
            elif self.ends("ent"): pass
            else: return
        elif self.b[self.k - 1] == 'o':
            if self.ends("ion") and (self.b[self.j] == 's' or self.b[self.j] == 't'): pass
            elif self.ends("ou"): pass
            else: return
        elif self.b[self.k - 1] == 's':
            if self.ends("ism"): pass
            else: return
        elif self.b[self.k - 1] == 't':
            if self.ends("ate"): pass
            elif self.ends("iti"): pass
            else: return
        elif self.b[self.k - 1] == 'u':
            if self.ends("ous"): pass
            else: return
        elif self.b[self.k - 1] == 'v':
            if self.ends("ive"): pass
            else: return
        elif self.b[self.k - 1] == 'z':
            if self.ends("ize"): pass
            else: return
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k-1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k = self.k -1

    def stem(self, w):
        """Stem a word using Porter algorithm"""
        w = w.lower()
        self.b = w
        self.k = len(w) - 1
        self.k0 = 0
        if self.k > self.k0 + 1:
            self.step1ab()
            self.step1c()
            self.step2()
            self.step3()
            self.step4()
            self.step5()
        return self.b[:self.k+1]


class HTMLTextExtractor(HTMLParser):
    """Extract text and important words from HTML"""
    
    def __init__(self):
        super().__init__()
        self.text = []
        self.important_words = []
        self.current_tag = None
        self.in_title = False
        self.in_important = False
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag == 'title':
            self.in_title = True
        elif tag in ['b', 'strong', 'h1', 'h2', 'h3']:
            self.in_important = True
    
    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False
        elif tag in ['b', 'strong', 'h1', 'h2', 'h3']:
            self.in_important = False
        self.current_tag = None
    
    def handle_data(self, data):
        text = data.strip()
        if text:
            self.text.append(text)
            if self.in_title or self.in_important:
                self.important_words.append(text)
    
    def get_text(self):
        return ' '.join(self.text)
    
    def get_important_text(self):
        return ' '.join(self.important_words)


class SearchEngine:
    """Complete search engine with indexing and retrieval"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.stemmer = PorterStemmer()
        
        # Inverted index: {term: {doc_id: (tf, is_important)}}
        self.index = defaultdict(lambda: defaultdict(lambda: [0, False]))
        
        # Document metadata
        self.doc_urls = {}  # {doc_id: url}
        self.doc_lengths = {}  # {doc_id: number_of_terms}
        
        # Collection statistics
        self.num_docs = 0
        self.avg_doc_length = 0
        
    def tokenize(self, text: str) -> List[str]:
        """Extract alphanumeric tokens from text"""
        # Extract all alphanumeric sequences
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return tokens
    
    def process_document(self, content: str) -> Tuple[List[str], Set[str]]:
        """Extract tokens and important tokens from HTML"""
        parser = HTMLTextExtractor()
        try:
            parser.feed(content)
        except:
            # If parsing fails, treat as plain text
            return self.tokenize(content), set()
        
        # Get all text and important text
        all_text = parser.get_text()
        important_text = parser.get_important_text()
        
        # Tokenize
        tokens = self.tokenize(all_text)
        important_tokens = set(self.tokenize(important_text))
        
        return tokens, important_tokens
    
    def build_index(self):
        """Build inverted index from all JSON files"""
        print("Building index...")
        start_time = time.time()
        
        doc_id = 0
        total_length = 0
        
        # Traverse all JSON files
        for root, dirs, files in os.walk(self.data_path):
            for filename in files:
                if filename.endswith('.json'):
                    filepath = os.path.join(root, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        url = data.get('url', '')
                        content = data.get('content', '')
                        
                        if not content:
                            continue
                        
                        # Process document
                        tokens, important_tokens = self.process_document(content)
                        
                        # Store document metadata
                        self.doc_urls[doc_id] = url
                        self.doc_lengths[doc_id] = len(tokens)
                        total_length += len(tokens)
                        
                        # Build term frequency map
                        term_freq = defaultdict(int)
                        for token in tokens:
                            stemmed = self.stemmer.stem(token)
                            term_freq[stemmed] += 1
                        
                        # Identify important terms
                        important_stems = set()
                        for token in important_tokens:
                            stemmed = self.stemmer.stem(token)
                            important_stems.add(stemmed)
                        
                        # Add to index
                        for term, freq in term_freq.items():
                            is_important = term in important_stems
                            self.index[term][doc_id] = [freq, is_important]
                        
                        doc_id += 1
                        
                        if doc_id % 100 == 0:
                            print(f"Processed {doc_id} documents...")
                            
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
                        continue
        
        self.num_docs = doc_id
        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0
        
        elapsed = time.time() - start_time
        print(f"\nIndex built successfully!")
        print(f"Total documents: {self.num_docs}")
        print(f"Unique terms: {len(self.index)}")
        print(f"Average document length: {self.avg_doc_length:.2f} terms")
        print(f"Time elapsed: {elapsed:.2f} seconds")
    
    def save_index(self, filepath: str):
        """Save index to disk"""
        print(f"\nSaving index to {filepath}...")
        # Convert nested defaultdicts to regular dicts for pickling
        index_dict = {}
        for term, postings in self.index.items():
            index_dict[term] = dict(postings)
        
        data = {
            'index': index_dict,
            'doc_urls': self.doc_urls,
            'doc_lengths': self.doc_lengths,
            'num_docs': self.num_docs,
            'avg_doc_length': self.avg_doc_length
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print("Index saved successfully!")
    
    def load_index(self, filepath: str):
        """Load index from disk"""
        print(f"Loading index from {filepath}...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct index with defaultdict structure
        self.index = defaultdict(lambda: defaultdict(lambda: [0, False]))
        for term, postings in data['index'].items():
            for doc_id, values in postings.items():
                self.index[term][doc_id] = values
        
        self.doc_urls = data['doc_urls']
        self.doc_lengths = data['doc_lengths']
        self.num_docs = data['num_docs']
        self.avg_doc_length = data['avg_doc_length']
        print("Index loaded successfully!")
    
    def compute_score(self, query_terms: List[str], k1: float = 1.5, b: float = 0.75) -> Dict[int, float]:
        """
        Compute BM25 scores with importance weighting
        
        BM25 formula with modifications:
        - Standard BM25 for term frequency and document length normalization
        - Boosting factor for important words (2x weight)
        """
        scores = defaultdict(float)
        
        # Get document frequency for each query term
        for term in query_terms:
            if term not in self.index:
                continue
            
            # Document frequency
            df = len(self.index[term])
            
            # IDF computation (standard BM25)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            # Score each document containing this term
            for doc_id, (tf, is_important) in self.index[term].items():
                # BM25 term frequency component
                doc_len = self.doc_lengths[doc_id]
                normalized_tf = tf / (1.0 - b + b * (doc_len / self.avg_doc_length))
                score = idf * (tf * (k1 + 1.0)) / (tf + k1 * normalized_tf)
                
                # Boost important terms
                if is_important:
                    score *= 2.0
                
                scores[doc_id] += score
        
        return scores
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for documents matching query"""
        start_time = time.time()
        
        # Tokenize and stem query
        tokens = self.tokenize(query)
        query_terms = [self.stemmer.stem(token) for token in tokens]
        
        if not query_terms:
            return []
        
        # Compute scores
        scores = self.compute_score(query_terms)
        
        # Sort by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = [(self.doc_urls[doc_id], score) for doc_id, score in ranked_docs]
        
        elapsed = time.time() - start_time
        print(f"Search completed in {elapsed*1000:.2f}ms")
        
        return results


def main():
    """Main function for building index and running search interface"""
    
    # Configuration
    DATA_PATH = './ANALYST'
    INDEX_PATH = './search_index.pkl'
    
    # Initialize search engine
    engine = SearchEngine(DATA_PATH)
    
    # Check if index exists
    if os.path.exists(INDEX_PATH):
        print("Found existing index.")
        engine.load_index(INDEX_PATH)
    else:
        print("No existing index found. Building new index...")
        engine.build_index()
        engine.save_index(INDEX_PATH)
    
    # Interactive search interface
    print("\n" + "="*70)
    print("SEARCH ENGINE - Information Analyst Track")
    print("="*70)
    print(f"Indexed {engine.num_docs} documents")
    print("Enter your search queries (or 'quit' to exit)")
    print("="*70 + "\n")
    
    while True:
        query = input("Search: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        # Perform search
        results = engine.search(query, top_k=10)
        
        # Display results
        print(f"\nFound {len(results)} results:\n")
        for i, (url, score) in enumerate(results, 1):
            print(f"{i}. {url}")
            print(f"   Score: {score:.4f}\n")
        
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()

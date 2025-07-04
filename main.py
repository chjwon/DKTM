import os
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

from keyword_extract import get_KeyBert_result, get_tf_idf_result

# Import topic modeling functions
from topic_model import get_BERTopic_result, get_Top2vec_result

class DKTM:    
    def __init__(self, documents_dir: str, output_dir: str = "./output"):
        """
        Args:
            documents_dir: Directory containing documents to process
            output_dir: Directory to save output CSV files
        """
        self.documents_dir = Path(documents_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for documents and results
        self.documents = []
        self.file_names = []
        self.keywords_results = []
        self.topic_results = {}
        
    def load_documents(self) -> None:
        print("Loading documents...")
        
        # Support common text file extensions - will be added
        text_extensions = ['.txt', '.md', '.doc', '.docx']
        
        for file_path in self.documents_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # Only add non-empty documents
                            self.documents.append(content)
                            self.file_names.append(file_path.name)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
        
        print(f"Loaded {len(self.documents)} documents")
        
        if len(self.documents) == 0:
            raise ValueError("No documents found in the specified directory")
    
    def extract_keywords(self, method: str = "keybert") -> None:
        """
        Args:
            method: Keyword extraction method ('none', 'keybert', 'tf-idf')
        """
        print(f"Extracting keywords using {method}...")
        
        if method.lower() == "none":
            # No keyword extraction, use original documents
            self.keywords_results = [[doc] for doc in self.documents]
        elif method.lower() == "keybert":
            self.keywords_results = get_KeyBert_result(self.documents)
        elif method.lower() == "tf-idf":
            self.keywords_results = get_tf_idf_result(self.documents)
        else:
            raise ValueError(f"Unknown keyword extraction method: {method}")
        
        print(f"Keyword extraction completed for {len(self.keywords_results)} documents")
    
    def save_keywords_csv(self, method: str) -> str:
        """        
        Args:
            method: Keyword extraction method used
        Returns:
            Path to saved CSV file
        """
        csv_path = self.output_dir / f"keywords_{method}.csv"
        
        print(f"Saving keywords to {csv_path}...")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # write header
            max_keywords = max(len(keywords) for keywords in self.keywords_results) if self.keywords_results else 0
            header = ['file_name'] + [f'keyword_{i+1}' for i in range(max_keywords)]
            writer.writerow(header)
            
            # write data
            for file_name, keywords in zip(self.file_names, self.keywords_results):
                row = [file_name]
                
                # handle different keyword formats
                if isinstance(keywords, list):
                    if len(keywords) > 0 and isinstance(keywords[0], tuple):
                        # Format: [(keyword, score), ...]
                        keyword_strings = [kw[0] for kw in keywords]
                    else:
                        # Format: [keyword, ...] or [" "]
                        keyword_strings = keywords
                else:
                    keyword_strings = [str(keywords)]
                
                # add keywords to row
                row.extend(keyword_strings)
                
                # pad with empty strings if needed
                while len(row) < len(header):
                    row.append('')
                
                writer.writerow(row)
        
        print(f"Keywords saved to {csv_path}")
        return str(csv_path)
    
    def perform_topic_modeling(self, topic_method: str = "bertopic", 
                             emb_method: str = "word2vec") -> None:
        """        
        Args:
            topic_method: Topic modeling method ('bertopic', 'top2vec')
            emb_method: Embedding method ('doc2vec', 'GPT2', 'word2vec')
        """
        print(f"Performing topic modeling using {topic_method} with {emb_method}...")
        
        if topic_method.lower() == "bertopic":
            self.topic_results = get_BERTopic_result(self.documents, emb_method)
        elif topic_method.lower() == "top2vec":
            self.topic_results = get_Top2vec_result(self.documents, emb_method)
        else:
            raise ValueError(f"Unknown topic modeling method: {topic_method}")
        
        print("Topic modeling completed")
    
    def save_topic_results_csv(self, topic_method: str, emb_method: str) -> Tuple[str, str]:
        """
        Save topic modeling results to CSV files
        
        Args:
            topic_method: Topic modeling method used
            emb_method: Embedding method used
            
        Returns:
            Tuple of (clusters_csv_path, assignments_csv_path)
        """
        # save cluster topics
        clusters_csv_path = self.output_dir / f"clusters_{topic_method}_{emb_method}.csv"
        assignments_csv_path = self.output_dir / f"assignments_{topic_method}_{emb_method}.csv"
        
        print(f"Saving topic results to {clusters_csv_path} and {assignments_csv_path}...")
        
        # extract cluster information and document assignments
        clusters_info = self.topic_results.get('clusters', {})
        document_assignments = self.topic_results.get('assignments', [])
        
        # save clusters and their topic words
        with open(clusters_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # write header
            max_topic_words = max(len(words) for words in clusters_info.values()) if clusters_info else 0
            header = ['cluster_name'] + [f'topic_word_{i+1}' for i in range(max_topic_words)]
            writer.writerow(header)
            
            # write cluster data
            for cluster_name, topic_words in clusters_info.items():
                row = [cluster_name] + list(topic_words)
                # pad with empty strings if needed
                while len(row) < len(header):
                    row.append('')
                writer.writerow(row)
        
        # save document-cluster assignments
        with open(assignments_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file_name', 'cluster_name'])
            
            for file_name, cluster_name in zip(self.file_names, document_assignments):
                writer.writerow([file_name, cluster_name])
        
        print(f"Topic results saved to {clusters_csv_path} and {assignments_csv_path}")
        return str(clusters_csv_path), str(assignments_csv_path)
    
    def run_pipeline(self, keyword_method: str = "keybert", 
                    topic_method: str = "bertopic", 
                    emb_method: str = "word2vec") -> Dict[str, str]:
        """        
        Args:
            keyword_method: Keyword extraction method
            topic_method: Topic modeling method
            emb_method: Embedding method
            
        Returns:
            Dictionary with paths to output files
        """
        print("Starting DKTM pipeline...")
        print(f"Configuration: keyword={keyword_method}, topic={topic_method}, embedding={emb_method}")
        
        # 1) Load documents
        self.load_documents()
        
        # 2) Extract keywords
        self.extract_keywords(keyword_method)
        
        # 3) Save keywords
        keywords_csv = self.save_keywords_csv(keyword_method)
        
        # 4) Perform topic modeling
        self.perform_topic_modeling(topic_method, emb_method)
        
        # 5) Save topic results
        clusters_csv, assignments_csv = self.save_topic_results_csv(topic_method, emb_method)
        
        results = {
            'keywords_csv': keywords_csv,
            'clusters_csv': clusters_csv,
            'assignments_csv': assignments_csv
        }
        
        print("DKTM pipeline completed successfully!")
        print(f"Output files saved in: {self.output_dir}")
        
        return results


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='DKTM: Document Keyword Topic Modeling')
    
    parser.add_argument('documents_dir', type=str, 
                       help='Directory containing documents to process')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for CSV files (default: ./output)')
    parser.add_argument('--keyword_method', type=str, default='keybert',
                       choices=['none', 'keybert', 'tf-idf'],
                       help='Keyword extraction method (default: keybert)')
    parser.add_argument('--topic_method', type=str, default='bertopic',
                       choices=['bertopic', 'top2vec'],
                       help='Topic modeling method (default: bertopic)')
    parser.add_argument('--emb_method', type=str, default='word2vec',
                       choices=['doc2vec', 'GPT2', 'word2vec'],
                       help='Embedding method (default: word2vec)')
    
    args = parser.parse_args()
    
    try:
        # Initialize DKTM system
        dktm = DKTM(args.documents_dir, args.output_dir)
        
        # Run pipeline
        results = dktm.run_pipeline(
            keyword_method=args.keyword_method,
            topic_method=args.topic_method,
            emb_method=args.emb_method
        )
        
        print("\nGenerated files:")
        for key, path in results.items():
            print(f"  {key}: {path}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
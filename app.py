import os
import json
import datetime
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import re
import random

class ContentAITrainer:
    def __init__(self):
        self.training_data_dir = "training_data"
        os.makedirs(self.training_data_dir, exist_ok=True)
        self.sentence_patterns = {}  # Store complete sentences with their contexts
        self.keyword_index = {}      # Index sentences by keywords
        self.initialize_with_existing_data()
        
    def initialize_with_existing_data(self):
        try:
            training_files = [f for f in os.listdir(self.training_data_dir) 
                            if f.endswith('.txt')]
            
            if training_files:
                print("Found existing training data. Initializing system...")
                all_content = self.load_all_training_data()
                if all_content.strip():
                    self.train_model(all_content)
                    print("System initialized with existing training data!")
                    return True
            return False
        except Exception as e:
            print(f"Error initializing with existing data: {str(e)}")
            return False
    
    def save_training_data(self, content, source_type, metadata=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if source_type == "text":
            filename = f"text_input_{timestamp}.txt"
        elif source_type == "url":
            filename = f"url_content_{timestamp}.txt"
        elif source_type == "file":
            filename = f"file_content_{timestamp}.txt"
        else:
            raise ValueError("Invalid source type")
            
        filepath = os.path.join(self.training_data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            if metadata:
                f.write(f"Metadata:\n{json.dumps(metadata, indent=2)}\n\nContent:\n")
            f.write(content)
        return filepath

    def load_all_training_data(self):
        all_content = []
        for filename in os.listdir(self.training_data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(self.training_data_dir, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "Metadata:" in content:
                        content = content.split("Content:\n")[-1]
                    all_content.append(content)
        return '\n'.join(all_content)

    def extract_keywords(self, text):
        # Remove common words and keep meaningful keywords
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        words = text.lower().split()
        return [w for w in words if w not in common_words and len(w) > 2]

    def preprocess_text(self, text):
        # Split into sentences and clean
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]  # Only keep meaningful sentences
        return sentences

    def find_sentence_context(self, sentences, index):
        # Get surrounding sentences as context
        start = max(0, index - 1)
        end = min(len(sentences), index + 2)
        return sentences[start:end]

    def train_model(self, content):
        try:
            # Reset existing patterns
            self.sentence_patterns.clear()
            self.keyword_index.clear()
            
            # Process the content
            sentences = self.preprocess_text(content)
            
            # Build patterns with context
            for i, sentence in enumerate(sentences):
                # Get context (previous and next sentences)
                context = self.find_sentence_context(sentences, i)
                
                # Extract keywords from the sentence
                keywords = self.extract_keywords(sentence)
                
                # Store the sentence and its context
                pattern_key = ' '.join(keywords)
                if pattern_key not in self.sentence_patterns:
                    self.sentence_patterns[pattern_key] = []
                self.sentence_patterns[pattern_key].append({
                    'sentence': sentence,
                    'context': context
                })
                
                # Index each keyword
                for keyword in keywords:
                    if keyword not in self.keyword_index:
                        self.keyword_index[keyword] = set()
                    self.keyword_index[keyword].add(pattern_key)
            
            print(f"Trained on {len(sentences)} sentences")
            print(f"Learned {len(self.sentence_patterns)} patterns")
            return True
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False

    def find_relevant_content(self, prompt):
        # Extract keywords from prompt
        prompt_keywords = self.extract_keywords(prompt)
        
        # Find patterns containing prompt keywords
        relevant_patterns = set()
        for keyword in prompt_keywords:
            if keyword in self.keyword_index:
                relevant_patterns.update(self.keyword_index[keyword])
        
        # Collect relevant sentences with their context
        relevant_content = []
        for pattern in relevant_patterns:
            relevant_content.extend(self.sentence_patterns[pattern])
        
        return relevant_content

    def generate_content(self, prompt):
        if not self.sentence_patterns:
            print("System not trained yet! Please add training data first.")
            return None
        
        try:
            # Find relevant content
            relevant_content = self.find_relevant_content(prompt)
            
            if not relevant_content:
                print("No relevant content found in training data.")
                return None
            
            # Select the most relevant sentences based on keyword matches
            prompt_keywords = set(self.extract_keywords(prompt))
            
            # Score and sort content by relevance
            scored_content = []
            for content in relevant_content:
                sentence = content['sentence']
                sentence_keywords = set(self.extract_keywords(sentence))
                score = len(prompt_keywords.intersection(sentence_keywords))
                scored_content.append((score, content))
            
            # Filter out zero-scored content
            scored_content = [sc for sc in scored_content if sc[0] > 0]
            scored_content.sort(reverse=True, key=lambda x: x[0])  # Sort by score
            
            # Generate response using top relevant content
            response_parts = []
            used_sentences = set()
            
            for _, content in scored_content[:3]:  # Use top 3 most relevant pieces
                if content['sentence'] not in used_sentences:
                    response_parts.extend(content['context'])
                    used_sentences.update(content['context'])
            
            # Remove duplicates while preserving order
            response_parts = list(dict.fromkeys(response_parts))
            
            # Combine into final response
            response = ' '.join(response_parts)
            
            return response, None
            
        except Exception as e:
            print(f"Error generating content: {str(e)}")
            return None

def main_menu():
    trainer = ContentAITrainer()
    while True:
        print("\nContent AI Menu:")
        print("1. Add training text")
        print("2. Add URL content")
        print("3. Add file content")
        print("4. Generate content")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            text = input("Enter training text: ")
            filepath = trainer.save_training_data(text, "text", {"type": "user_input"})
            print(f"Text saved to {filepath}")
            
            if trainer.train_model(text):
                print("Model trained successfully!")
            else:
                print("Error occurred during training.")
        
        elif choice == '2':
            url = input("Enter URL: ")
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
                
                filepath = trainer.save_training_data(text, "url", {"type": "url_content", "source": url})
                print(f"URL content saved to {filepath}")
                
                if trainer.train_model(text):
                    print("Model trained successfully with URL content!")
                else:
                    print("Error occurred during training.")
            except Exception as e:
                print(f"Error processing URL: {str(e)}")
        
        elif choice == '3':
            file_path = input("Enter file path (txt/csv/json/xml): ")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    filepath = trainer.save_training_data(content, "file", {
                        "type": "file_content",
                        "source_file": file_path,
                        "file_type": file_path.split('.')[-1]
                    })
                    print(f"File content saved to {filepath}")
                    
                    if trainer.train_model(content):
                        print("Model trained successfully with file content!")
                    else:
                        print("Error occurred during training.")
                except Exception as e:
                    print(f"Error processing file: {str(e)}")
            else:
                print("File not found!")
        
        elif choice == '4':
            if not trainer.sentence_patterns:
                print("System not trained yet! Please add training data using options 1, 2, or 3 first.")
                continue
                
            prompt = input("Enter your prompt: ")
            result = trainer.generate_content(prompt)
            
            if result:
                content, _ = result
                print("\nGenerated Content:")
                print("-" * 50)
                print(content)
                print("-" * 50)
            else:
                print("Could not find relevant information in the training data.")
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
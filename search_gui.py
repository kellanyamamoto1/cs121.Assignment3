import tkinter as tk
from tkinter import ttk, scrolledtext
import webbrowser
from search_engine import SearchEngine


class SearchEngineGUI:
    """GUI interface for the search engine"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("UCI Search Engine - Information Analyst Track")
        self.root.geometry("900x700")
        
        # Initialize search engine
        self.engine = SearchEngine('/home/claude/ANALYST')
        try:
            self.engine.load_index('/home/claude/search_index.pkl')
            self.status_text = f"Index loaded: {self.engine.num_docs} documents indexed"
        except:
            self.status_text = "Error: Could not load index"
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""

        title_frame = tk.Frame(self.root, bg="#2196F3", height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="UCI Search Engine",
            font=("Arial", 24, "bold"),
            bg="#2196F3",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Search frame
        search_frame = tk.Frame(self.root, bg="white", padx=20, pady=20)
        search_frame.pack(fill=tk.X)
        
        search_label = tk.Label(
            search_frame,
            text="Search Query:",
            font=("Arial", 12),
            bg="white"
        )
        search_label.pack(anchor=tk.W)
        
        search_entry_frame = tk.Frame(search_frame, bg="white")
        search_entry_frame.pack(fill=tk.X, pady=10)
        
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(
            search_entry_frame,
            textvariable=self.search_var,
            font=("Arial", 14),
            relief=tk.SOLID,
            borderwidth=1
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.search_entry.bind('<Return>', lambda e: self.perform_search())
        
        search_button = tk.Button(
            search_entry_frame,
            text="Search",
            command=self.perform_search,
            bg="#2196F3",
            fg="white",
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            padx=30,
            cursor="hand2"
        )
        search_button.pack(side=tk.LEFT, padx=(10, 0))
        status_frame = tk.Frame(self.root, bg="#f5f5f5", height=30)
        status_frame.pack(fill=tk.X)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text=self.status_text,
            font=("Arial", 9),
            bg="#f5f5f5",
            fg="#666666",
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=20)
        
        # Results frame
        results_frame = tk.Frame(self.root, bg="white")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        results_label = tk.Label(
            results_frame,
            text="Search Results",
            font=("Arial", 14, "bold"),
            bg="white",
            anchor=tk.W
        )
        results_label.pack(fill=tk.X, pady=(0, 10))
        
        # Results text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            font=("Arial", 11),
            wrap=tk.WORD,
            relief=tk.SOLID,
            borderwidth=1,
            state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        self.results_text.tag_configure("url", foreground="#2196F3", underline=True)
        self.results_text.tag_configure("score", foreground="#666666", font=("Arial", 9))
        self.results_text.tag_configure("number", foreground="#FF9800", font=("Arial", 11, "bold"))
    
        self.results_text.tag_bind("url", "<Button-1>", self.open_url)
        self.results_text.tag_bind("url", "<Enter>", lambda e: self.results_text.config(cursor="hand2"))
        self.results_text.tag_bind("url", "<Leave>", lambda e: self.results_text.config(cursor=""))
        self.search_entry.focus()

    #need to add perform search and open_url functions here

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SearchEngineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
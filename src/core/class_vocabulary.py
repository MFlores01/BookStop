class ClassVocab:
    """Class containing the dictionaries for each book task classification"""

    @staticmethod
    def book_availability_vocab():
        """Dictionary for book availability classification"""
        vocab = ["available", "in stock", "do you have", "availability"]
        return vocab
    
    @staticmethod
    def book_recommendation_vocab():
        """Dictionary for book recommendation classification"""
        vocab = ["AI", "medical", "coding", "sport", "give me a book", "Find", "recommend", "suggest", "good book", "any book", "kids", "novel", "books for kids"]
        return vocab
    
    @staticmethod
    def book_rent_vocab():
        """Dictionary for book rental classification"""
        vocab = ["to avail", "take", "rent", "borrow", "lend", "can i get", "check out", "issue book"]
        return vocab
    
    @staticmethod
    def book_return_vocab():
        """Dictionary for book return classification"""
        vocab = ["how to return a book", "return", "give back", "bring back", "returning", "drop off"]
        return vocab
    
    @staticmethod
    def book_talk_vocab():
        """Dictionary for book discussion classification"""
        vocab = ["tell me", "glaze", "fire", "tea", "emotion", "disgusting", "dont like", "not so much", "has", "pretty", "meh", "hate", "Talk", "know", "learn", "amazed", "love", "like", "chat", "think", "analyze", "thoughts", "discussion", "talk about", "opinions", "review"]
        return vocab
    
    @staticmethod
    def general_vocab():
        """Dictionary for general classification"""
        vocab = ["como esta", "slay", "skrrt", "skibidi", "hi", "hello", "hey", "how are you", "good morning", "good afternoon", "good evening"]
        return vocab
    
    @staticmethod 
    def not_book_related_vocab():
        vocab = ["salary", "pay", "work", "job", "employee", "staff", "cloudstaff", "address", "lloyd", "joy", "password", "code", "python"]
        return vocab
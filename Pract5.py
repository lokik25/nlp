# Simple Morphological analyzer
def add_suffix(word, suffix):
    return word + suffix

def delete_suffix(word, suffix):
    if word.endswith(suffix):
        return word[:-len(suffix)]
    else:
        return word

# Examples
print("Add -ed to 'play':", add_suffix('play', 'ed'))
print("Delete -y from 'happy':", delete_suffix('happy', 'y'))

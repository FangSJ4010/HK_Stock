from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, NLTKTextSplitter


'''按固定长度分块'''
def Split_Character(data):
    split = CharacterTextSplitter(
        separator="\n",          # 按换行符分割
        chunk_size=512,          # 每个块最多1000字符
        chunk_overlap=100,        # 块间重叠150字符
        length_function=len,      # 使用len计算长度
        is_separator_regex=False, # 分隔符不是正则
        keep_separator=True,      # 保留换行符在内容中
        strip_whitespace=True     # 去除首尾空白
    )
    return split.split_documents(data)


'''递归字符分块'''
def Recursive_split(data):
    split = RecursiveCharacterTextSplitter(
        chunk_size=512,  # 块大小
        chunk_overlap=100,  # 块重叠
        length_function=len,  # 长度函数（按字符）
        separators=["\n\n", "\n", " ", "。"] # 可选，修改默认分隔符
    )
    return split.split_documents(data)


'''按语义感知分块'''
def Split_NLTK(data):
    split = NLTKTextSplitter(
        chunk_size=512,
        chunk_overlap=100
    )
    return split.split_documents(data)


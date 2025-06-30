import time
import multiprocessing
from datasketch import MinHash, MinHashLSH


def shingle_document(text, shingle_size=13):
    """Generate word-based shingles from a document."""
    # Split the text into words
    words = text.split()
    # Generate shingles that are sequences of 'shingle_size' consecutive words
    shingles = set(
        " ".join(words[i : i + shingle_size])
        for i in range(len(words) - shingle_size + 1)
    )
    return shingles


m = MinHash(num_perm=128)
perm = m.permutations


def create_minhash(shingles, num_perm=128):
    """Create a MinHash object from the set of shingles."""
    m = MinHash(permutations=perm)
    m.update_batch(map(lambda x: x.encode("utf-8"), shingles))
    # for shingle in shingles:
    #     m.update(shingle.encode('utf-8'))
    return m


def abstein_string_for_decon(string):
    # Abstein the reading comprehension subject in MMLU where a paragraph from Wikipedia is given in the question
    return "refers to the following information" in string


def remove_duplicates_with_minhash(
    documents, string_for_decontamination=None, threshold=0.8, num_perm=128
):
    # Apply 13-gram Jaccard similarity deduplication and removes ones with similarity > 80% compared to former docs.
    # Remove chunks shorter than 13 words.

    # Create an LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Dictionary to store the MinHash of each document
    minhashes = {}

    # Hash string for decontamination first so contaminated samples will be removed
    decon_offset = 0
    if string_for_decontamination is not None and not abstein_string_for_decon(
        string_for_decontamination
    ):
        shingles = shingle_document(string_for_decontamination)
        m_decon = create_minhash(shingles, num_perm)
        lsh.insert(f"doc_{decon_offset}", m_decon)
        minhashes[decon_offset] = m_decon
        decon_offset = 1

    # Populate the LSH index
    short_chunk_indices = []
    for idx, ctx in enumerate(documents, start=decon_offset):
        doc = ctx["retrieval text"]
        shingles = shingle_document(doc)
        if not shingles:
            short_chunk_indices.append(idx - decon_offset)
        m = create_minhash(shingles, num_perm)
        lsh.insert(f"doc_{idx}", m)
        minhashes[idx] = m

    # List to keep track of non-duplicate document indices
    non_duplicate_indices = []

    # Check each document against the LSH index
    for idx, m in minhashes.items():
        if idx < decon_offset:
            continue

        # Query the LSH for near-duplicate candidates
        result = lsh.query(m)

        # print(result)
        # print([minhashes[int(doc_id.split("_")[1])].jaccard(m) for doc_id in result])

        # If the document is the only one in its bucket or it appears first in the list
        if all(
            minhashes[int(doc_id.split("_")[1])].jaccard(m) <= threshold
            or int(doc_id.split("_")[1]) >= idx
            for doc_id in result
        ):
            non_duplicate_indices.append(idx - decon_offset)

    # Return non-duplicate documents
    deduplicated_documents = [
        documents[i] for i in non_duplicate_indices if i not in short_chunk_indices
    ]
    [doc.update({"quality score": 1}) for doc in deduplicated_documents]
    removed_documents = [doc for doc in documents if doc not in deduplicated_documents]
    [doc.update({"quality score": 0}) for doc in removed_documents]

    print(f"Non-deduplication ctxs num: {len(deduplicated_documents)}")
    # for c in deduplicated_documents:
    #     try:
    #         print(c['retrieval text'][:10])
    #     except:
    #         print(c)
    # if len(deduplicated_documents[0]['retrieval text'].split(' ')) < 13:
    #     import pdb; pdb.set_trace()
    return deduplicated_documents  # + removed_documents


def process_item(data_item):
    time.sleep(0.0001)
    id_, ex = data_item
    ex["ctxs"] = remove_duplicates_with_minhash(
        ex["ctxs"], string_for_decontamination=ex["raw_query"]
    )
    return id_, ex


def multiprocess_deduplication(data):
    items_to_process = list(enumerate(data))
    pool = multiprocessing.Pool(processes=32)
    for result in pool.imap(process_item, items_to_process):
        id_, updated_ex = result
        data[id_] = updated_ex
    return data


if __name__ == "__main__":
    # Example usage:
    question = (
        "Answer these questions:\n\nQ: when did the eagles win last super bowl?\nA:"
    )
    docs = [
        "Eagles won the Super Bowl.",
        "Machine learning provides the ability to automatically learn and improve from experience without being explicitly programmed."
        * 20,
        "Machine learning provides the ability to automatically learn and improve from experience without being explicitly programmed."
        * 20
        + ".",
        "An entirely different document looks nothing like the others and should not be considered a duplicate."
        * 20,
        "Short sentence." * 1,
        "As someone who lived in Philly for about five years, I agree about the city\u2019s greatness \u2014 which makes the juxtaposition between its friendly day-to-day interactions and sometimes psychotic sports fandom even more jarring. The Eagles did win three NFL championships before the Super Bowl existed, most recently in 1960. But any fan who was following the team back then is now at least into their mid-60s, if not much older. It is, to say the least, a distant memory from another era. Granted, the Sixers went on their infamous tanking expedition during this span.",
    ] * 1
    import time

    num_ex = 1

    start = time.time()
    data1 = []
    for _ in range(num_ex):
        cleaned_ex = remove_duplicates_with_minhash(
            [{"retrieval text": doc} for doc in docs], question
        )
        data1.append(cleaned_ex)
    time1 = time.time() - start

    # ori_data = [{'raw_query': docs[0], 'ctxs': [{'retrieval text': doc} for doc in docs]}] * num_ex
    # start = time.time()
    # data2 = multiprocess_deduplication(ori_data)
    # time2 = time.time()-start

    # assert data2[0]['ctxs'] == data1[0]

    # print(time1)
    # print(time2)

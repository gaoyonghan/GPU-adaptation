from detection import require,integer,number

def validate_tables(data):
    require(isinstance(data, dict) and isinstance(data.get('tables'), list), 'tables list required')
    ids = set()
    for table in data['tables']:
        key = table.get('table_id')
        require(isinstance(key, str) and key and (key not in ids), 'unique string table_id required')
        ids.add(key)
        require(isinstance(table.get('coordinate_space'), str) and table['coordinate_space'], 'coordinate space required')
        cells = table.get('cells')
        require(isinstance(cells, list), 'cells must be a list, including empty predictions')
        for cell in cells:
            require(isinstance(cell.get('cell_text'), str), 'cell_text must be explicit, including empty text')
            bbox = cell.get('bbox')
            require(isinstance(bbox, list) and len(bbox) == 4 and all((number(x) for x in bbox)) and (bbox[2] >= bbox[0]) and (bbox[3] >= bbox[1]), 'cell bbox must be finite xyxy')
            for field in ('row_nums', 'column_nums'):
                values = cell.get(field)
                require(isinstance(values, list) and values and all((integer(x) and x >= 0 for x in values)), 'nonnegative span indices required')
                require(sorted(values) == list(range(min(values), max(values) + 1)), 'cell spans must contain unique contiguous indices')
    return ids

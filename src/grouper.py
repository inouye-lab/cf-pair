from wilds.common.grouper import Grouper


class ConditionalGrouper(Grouper):
    def __init__(self, dataset, group_by_fields)
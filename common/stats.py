from dataclasses import dataclass
import json
from typing import List, Dict


@dataclass
class Stat:
    title: str
    value: str
    description: str
    private: bool

    def __repr__(self):
        if self.private:
            return f"{self.title}: *********"
        else:       
            return f"{self.title}: {self.value}"

    def __str__(self):
        if self.private:
            return f"{self.title}: *********"
        else:
            return f"{self.title}: {self.value}"

    def to_dict(self):
        if self.private:
            return {
                "title": self.title,
                "value": "*********",
                "description": self.description,
                "private": True
            }
        else:
            return {
                "title": self.title,
                "value": self.value,
                "description": self.description,
                "private": False
            }

    def update(self, value: str):
        self.value = value
        if self.private:
            self.value = "*********"
        else:
            self.value = value

    def get_value(self):
        return self.value

    def get_description(self):
        return self.description


@dataclass
class Stats:
    group_name: str
    stats: List[Stat]

class Statistics:
    """Statistics for the system"""
    stats: List[Stats]
    group_stats: Dict[str, List[Stats]]

    def __init__(self):
        self.stats = []
        self.group_stats = {}

    def add_stat(self, group_name: str, stat: Stat):
        self.stats.append(Stats(group_name, [stat]))
        if group_name not in self.group_stats:
            self.group_stats[group_name] = []
        self.group_stats[group_name].append(stat)

    def get_stat(self, title: str):
        for stat in self.stats:
            if stat.title == title:
                return stat

    def get_stats(self, group_name: str):
        return self.group_stats[group_name]


statistics = Statistics()

def init_system_stats():

    statistics.add_stat("memory", Stat(title="total_memories", value="0", description="Total memories in the memory store", private=False))
    statistics.add_stat("memory", Stat(title="gpu_memories", value="0", description="GPU memories in the memory store", private=False))
    statistics.add_stat("memory", Stat(title="max_memories", value="0", description="Max memories in the memory store", private=False))
    statistics.add_stat("brain", Stat(title="total_interactions", value="0", description="Total interactions in the brain", private=False))
    statistics.add_stat("brain", Stat(title="positive_interactions", value="0", description="Positive interactions in the brain", private=False))
    statistics.add_stat("brain", Stat(title="negative_interactions", value="0", description="Negative interactions in the brain", private=False))
    statistics.add_stat("brain", Stat(title="empathetic_responses", value="0", description="Empathetic responses in the brain", private=False))
    statistics.add_stat("brain", Stat(title="informative_responses", value="0", description="Informative responses in the brain", private=False))
    statistics.add_stat("brain", Stat(title="gpu_operations", value="0", description="GPU operations in the brain", private=False))
    statistics.add_stat("brain", Stat(title="experiences_learned", value="0", description="Experiences learned in the brain", private=False))
    statistics.add_stat("brain", Stat(title="online_learning_updates", value="0", description="Online learning updates in the brain", private=False))
    return statistics


system_stats = init_system_stats()
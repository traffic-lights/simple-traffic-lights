from generators.random_generator import RandomGenerator
from generators.widget_generator import WidgetGenerator
from generators.sinusoidal_generator import SinusoidalGenerator
from generators.const_generator import ConstGenerator

GENERATORS_TYPE_MAPPER = {
    'const': ConstGenerator,
    'sin': SinusoidalGenerator,
    'widget': WidgetGenerator,
    'random': RandomGenerator
}

"""Country name → flag emoji mapping for TactiQ dashboard."""

_FLAGS: dict[str, str] = {
    'Afghanistan': '🇦🇫', 'Albania': '🇦🇱', 'Algeria': '🇩🇿', 'Angola': '🇦🇴',
    'Argentina': '🇦🇷', 'Armenia': '🇦🇲', 'Australia': '🇦🇺', 'Austria': '🇦🇹',
    'Azerbaijan': '🇦🇿', 'Bahrain': '🇧🇭', 'Belgium': '🇧🇪', 'Benin': '🇧🇯',
    'Bolivia': '🇧🇴', 'Bosnia and Herzegovina': '🇧🇦', 'Botswana': '🇧🇼',
    'Brazil': '🇧🇷', 'Bulgaria': '🇧🇬', 'Burkina Faso': '🇧🇫', 'Cameroon': '🇨🇲',
    'Canada': '🇨🇦', 'Cape Verde': '🇨🇻', 'Chile': '🇨🇱', 'China': '🇨🇳',
    'Colombia': '🇨🇴', 'Comoros': '🇰🇲', 'Congo DR': '🇨🇩', 'Costa Rica': '🇨🇷',
    'Croatia': '🇭🇷', 'Cuba': '🇨🇺', 'Czech Republic': '🇨🇿', 'Denmark': '🇩🇰',
    'Djibouti': '🇩🇯', 'Ecuador': '🇪🇨', 'Egypt': '🇪🇬', 'El Salvador': '🇸🇻',
    'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'Equatorial Guinea': '🇬🇶', 'Eritrea': '🇪🇷',
    'Estonia': '🇪🇪', 'Ethiopia': '🇪🇹', 'Finland': '🇫🇮', 'France': '🇫🇷',
    'Gabon': '🇬🇦', 'Gambia': '🇬🇲', 'Georgia': '🇬🇪', 'Germany': '🇩🇪',
    'Ghana': '🇬🇭', 'Greece': '🇬🇷', 'Guatemala': '🇬🇹', 'Guinea': '🇬🇳',
    'Guinea Bissau': '🇬🇼', 'Haiti': '🇭🇹', 'Honduras': '🇭🇳', 'Hungary': '🇭🇺',
    'Iceland': '🇮🇸', 'India': '🇮🇳', 'Indonesia': '🇮🇩', 'Iran': '🇮🇷',
    'Iraq': '🇮🇶', 'Ireland': '🇮🇪', 'Israel': '🇮🇱', 'Italy': '🇮🇹',
    'Ivory Coast': '🇨🇮', 'Jamaica': '🇯🇲', 'Japan': '🇯🇵', 'Jordan': '🇯🇴',
    'Kazakhstan': '🇰🇿', 'Kenya': '🇰🇪', 'Kosovo': '🇽🇰', 'Kuwait': '🇰🇼',
    'Latvia': '🇱🇻', 'Lebanon': '🇱🇧', 'Lesotho': '🇱🇸', 'Libya': '🇱🇾',
    'Lithuania': '🇱🇹', 'Luxembourg': '🇱🇺', 'Madagascar': '🇲🇬', 'Malaysia': '🇲🇾',
    'Mali': '🇲🇱', 'Malta': '🇲🇹', 'Mauritius': '🇲🇺', 'Mexico': '🇲🇽',
    'Montenegro': '🇲🇪', 'Morocco': '🇲🇦', 'Mozambique': '🇲🇿', 'Namibia': '🇳🇦',
    'Netherlands': '🇳🇱', 'New Zealand': '🇳🇿', 'Nigeria': '🇳🇬',
    'North Macedonia': '🇲🇰', 'Norway': '🇳🇴', 'Oman': '🇴🇲', 'Panama': '🇵🇦',
    'Paraguay': '🇵🇾', 'Peru': '🇵🇪', 'Philippines': '🇵🇭', 'Poland': '🇵🇱',
    'Portugal': '🇵🇹', 'Qatar': '🇶🇦', 'Romania': '🇷🇴', 'Russia': '🇷🇺',
    'Rwanda': '🇷🇼', 'Saudi Arabia': '🇸🇦', 'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'Senegal': '🇸🇳',
    'Serbia': '🇷🇸', 'Slovakia': '🇸🇰', 'Slovenia': '🇸🇮', 'Somalia': '🇸🇴',
    'South Africa': '🇿🇦', 'South Korea': '🇰🇷', 'South Sudan': '🇸🇸',
    'Spain': '🇪🇸', 'Sudan': '🇸🇩', 'Sweden': '🇸🇪', 'Switzerland': '🇨🇭',
    'Syria': '🇸🇾', 'Tanzania': '🇹🇿', 'Thailand': '🇹🇭', 'Togo': '🇹🇬',
    'Trinidad and Tobago': '🇹🇹', 'Tunisia': '🇹🇳', 'Turkey': '🇹🇷',
    'UAE': '🇦🇪', 'Uganda': '🇺🇬', 'Ukraine': '🇺🇦', 'United States': '🇺🇸',
    'Uruguay': '🇺🇾', 'Uzbekistan': '🇺🇿', 'Venezuela': '🇻🇪', 'Vietnam': '🇻🇳',
    'Wales': '🏴󠁧󠁢󠁷󠁬󠁳󠁿', 'Zambia': '🇿🇲', 'Zimbabwe': '🇿🇼',
}


def flag(country: str) -> str:
    """Return flag emoji for a country name, or '' if not found."""
    return _FLAGS.get(country, '')


def with_flag(country: str, sep: str = ' ') -> str:
    """Return 'FLAG Country' string, or just the country name if no flag found."""
    f = flag(country)
    return f'{f}{sep}{country}' if f else country

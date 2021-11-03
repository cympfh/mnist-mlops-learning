import streamlit


def table(ms: dict, title=None):
    """Simple key-value table"""
    if title:
        streamlit.caption(title)
    html = f"""
    <table><tbody>
        <tr><td></td><td>
        {'</td><td>'.join(map(str, ms.keys()))}
        </td></tr>
        <tr><td>value</td><td>
        {'</td><td>'.join(map(str, ms.values()))}
        </td></tr>
    </tbody></table>
    """
    streamlit.markdown(html, unsafe_allow_html=True)
    streamlit.write("")


def metrics(ms: dict, title=None):
    """Show multiple metrics in a row"""
    if title:
        streamlit.caption(title)
    if len(ms) == 0:
        streamlit.write("No metrics")
        return
    cols = streamlit.columns(len(ms))
    for i, (key, val) in enumerate(ms.items()):
        cols[i].metric(key, val)

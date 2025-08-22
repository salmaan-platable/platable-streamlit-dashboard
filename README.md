# Platable Streamlit Dashboard (Corrected)

- Uses your master file columns (date + time_of_order, order_state, order_number, order_value, delivery_fee, total, revenue, payout, brand, store_name, account_manager, phone_number, email, purchase_item_qty)
- **Vendor View** is hidden if your file has no vendor column.
- **Account Manager names** normalized to one of: `vinit`, `dante`, `platable`.
- **Popular time slots** use buckets: Morning (6–11), Afternoon (12–16), Evening (17–21), Other (22–5).
- **Pickup vs Delivery** inferred as: pickup when `delivery_fee <= 0`, else delivery.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

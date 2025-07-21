// PayStack Integration for Virtual Try-On App

interface PaystackConfig {
  key: string;
  email: string;
  amount: number; // in kobo (multiply by 100)
  currency?: string;
  ref?: string;
  callback?: (response: any) => void;
  onClose?: () => void;
  metadata?: {
    custom_fields?: Array<{
      display_name: string;
      variable_name: string;
      value: string;
    }>;
    [key: string]: any;
  };
}

interface CartItem {
  id: string;
  name: string;
  price: number;
  quantity: number;
  size?: string;
  color?: string;
}

export const PAYSTACK_PUBLIC_KEY = 'pk_live_044105bed5451b50cab46d9eb90ce9901fa3017b';

// Load PayStack script dynamically
export const loadPaystackScript = (): Promise<void> => {
  return new Promise((resolve, reject) => {
    if (window.PaystackPop) {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://js.paystack.co/v1/inline.js';
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load PayStack script'));
    document.head.appendChild(script);
  });
};

// Generate reference number
export const generateReference = (): string => {
  const timestamp = Date.now();
  const random = Math.floor(Math.random() * 1000000);
  return `vto_${timestamp}_${random}`;
};

// Calculate total amount in kobo
export const calculateTotal = (
  items: CartItem[], 
  shipping: number = 0, 
  tax: number = 0
): number => {
  const subtotal = items.reduce((sum, item) => sum + (item.price * item.quantity), 0);
  return Math.round((subtotal + shipping + tax) * 100); // Convert to kobo
};

// Initialize PayStack payment
export const initializePayment = async (config: PaystackConfig): Promise<void> => {
  try {
    await loadPaystackScript();
    
    const handler = window.PaystackPop.setup({
      key: config.key,
      email: config.email,
      amount: config.amount,
      currency: config.currency || 'GHS',
      ref: config.ref || generateReference(),
      callback: (response: any) => {
        console.log('Payment successful:', response);
        if (config.callback) {
          config.callback(response);
        }
      },
      onClose: () => {
        console.log('Payment modal closed');
        if (config.onClose) {
          config.onClose();
        }
      },
      metadata: config.metadata
    });

    handler.openIframe();
  } catch (error) {
    console.error('PayStack initialization error:', error);
    throw error;
  }
};

// Process cart checkout
export const processCheckout = async (
  items: CartItem[],
  customerEmail: string,
  customerInfo?: {
    firstName?: string;
    lastName?: string;
    phone?: string;
  },
  onSuccess?: (response: any) => void,
  onError?: (error: any) => void
): Promise<void> => {
  try {
    const subtotal = items.reduce((sum, item) => sum + (item.price * item.quantity), 0);
    const shipping = subtotal > 100 ? 0 : 10; // Free shipping over ₵100
    const tax = subtotal * 0.125; // 12.5% VAT
    const total = calculateTotal(items, shipping, tax);

    // Prepare metadata with order details
    const metadata = {
      order_id: generateReference(),
      customer_name: customerInfo?.firstName && customerInfo?.lastName 
        ? `${customerInfo.firstName} ${customerInfo.lastName}` 
        : 'Customer',
      items: items.map(item => ({
        id: item.id,
        name: item.name,
        price: item.price,
        quantity: item.quantity,
        size: item.size,
        color: item.color
      })),
      subtotal: subtotal,
      shipping: shipping,
      tax: tax,
      total: subtotal + shipping + tax,
      custom_fields: [
        {
          display_name: "Order Type",
          variable_name: "order_type",
          value: "Virtual Try-On Purchase"
        },
        {
          display_name: "Items Count",
          variable_name: "items_count",
          value: items.length.toString()
        }
      ]
    };

    await initializePayment({
      key: PAYSTACK_PUBLIC_KEY,
      email: customerEmail,
      amount: total,
      currency: 'GHS',
      metadata: metadata,
      callback: (response) => {
        // Payment successful
        if (onSuccess) {
          onSuccess(response);
        }
        
        // Here you would typically:
        // 1. Send the payment reference to your backend
        // 2. Verify the payment with PayStack
        // 3. Create order in your database
        // 4. Send confirmation email
        console.log('Payment completed successfully:', {
          reference: response.reference,
          amount: total / 100,
          items: items
        });
      },
      onClose: () => {
        console.log('Payment was cancelled');
      }
    });

  } catch (error) {
    console.error('Checkout process error:', error);
    if (onError) {
      onError(error);
    }
    throw error;
  }
};

// Verify payment (to be called from your backend)
export const verifyPayment = async (reference: string): Promise<any> => {
  // This should be done on your backend for security
  // Here's the structure for reference:
  
  const response = await fetch(`https://api.paystack.co/transaction/verify/${reference}`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer YOUR_SECRET_KEY`, // Use secret key on backend only
      'Content-Type': 'application/json'
    }
  });

  return response.json();
};

// Format amount for display
export const formatAmount = (amount: number): string => {
  return `₵${amount.toFixed(2)}`;
};

// Validate email for PayStack
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// Get payment status message
export const getPaymentStatusMessage = (status: string): string => {
  switch (status) {
    case 'success':
      return 'Payment completed successfully!';
    case 'failed':
      return 'Payment failed. Please try again.';
    case 'abandoned':
      return 'Payment was cancelled.';
    default:
      return 'Processing payment...';
  }
};

// Declare PayStack types for TypeScript
declare global {
  interface Window {
    PaystackPop: {
      setup: (config: any) => {
        openIframe: () => void;
      };
    };
  }
}